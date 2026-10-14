"""Multi-Sport Ingestion & Normalization Service.

Normalizes payloads from API-Sports across Football, Basketball, and Tennis,
maps status codes (e.g. Q1, Q2, S1, S2, FT, NS) into standard UI and database
representations, and upserts them into the universal Match SQLAlchemy model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import BookmakerOdds, Entity, Match, MatchStatus, Sport
from app.services.api_sports_client import APISportsClient, api_sports_client
from app.sofascore import (
    _get_or_create_entity,
    _get_or_create_sport,
    _insert_baseline_odds,
)

log = logging.getLogger("pikmatrix_multi_sport_service")


# --------------------------------------------------------------------------- #
# Status Standardization Matrix
# --------------------------------------------------------------------------- #
LIVE_STATUS_CODES = {
    # Football
    "1H", "2H", "HT", "ET", "BT", "P", "LIVE", "IN_PLAY", "INPROGRESS", "1ST_HALF", "2ND_HALF", "HALFTIME",
    # Basketball
    "Q1", "Q2", "Q3", "Q4", "OT", "1ST_QUARTER", "2ND_QUARTER", "3RD_QUARTER", "4TH_QUARTER",
    # Tennis
    "S1", "S2", "S3", "S4", "S5", "1ST_SET", "2ND_SET", "3RD_SET", "4TH_SET", "5TH_SET",
}

FINISHED_STATUS_CODES = {
    "FT", "AET", "PEN", "AOT", "POST", "FINAL", "FINISHED", "ENDED", "WO", "RET", "AWD",
}

CANCELLED_STATUS_CODES = {
    "PST", "CANC", "CANCELLED", "ABD", "ABANDONED", "SUSP", "SUSPENDED", "INT", "INTERRUPTED", "POSTPONED",
}

SCHEDULED_STATUS_CODES = {
    "NS", "TBD", "NOT_STARTED", "NOT STARTED", "SCHEDULED", "UPCOMING",
}


def standardize_match_status(
    sport: str,
    raw_status_code: Optional[str] = None,
    raw_long_status: Optional[str] = None,
    elapsed: Optional[Any] = None,
) -> Tuple[MatchStatus, str, bool]:
    """Standardize multi-sport status into (MatchStatus, display_period, is_live).

    Maps:
      - Basketball: 'Q1', 'Q2', 'Q3', 'Q4', 'OT' -> (MatchStatus.LIVE, 'Q1'..'OT', True)
      - Tennis: 'S1', 'S2', 'S3' -> (MatchStatus.LIVE, 'S1'..'S3', True)
      - Football: '1H', '2H', 'HT' -> (MatchStatus.LIVE, '1H'..'2H', True)
      - Finished: 'FT', 'AET', 'WO', 'RET' -> (MatchStatus.FINISHED, 'FT', False)
      - Scheduled: 'NS', 'TBD' -> (MatchStatus.SCHEDULED, 'NS', False)

    Returns:
        Tuple of (Enum MatchStatus, clean display string for live minute/quarter/set, is_live boolean)
    """
    code = (raw_status_code or "").strip().upper()
    long_desc = (raw_long_status or "").strip().upper()

    # 1. Direct short code inspection
    if code in LIVE_STATUS_CODES or "IN PLAY" in long_desc or "LIVE" in long_desc:
        display_label = code if code else ("LIVE" if not elapsed else f"{elapsed}'")
        return MatchStatus.LIVE, display_label, True

    if code in FINISHED_STATUS_CODES or "FINISHED" in long_desc or "FINAL" in long_desc or "ENDED" in long_desc:
        return MatchStatus.FINISHED, "FT", False

    if code in CANCELLED_STATUS_CODES or "CANCELLED" in long_desc or "POSTPONED" in long_desc:
        return MatchStatus.CANCELLED, "CANC", False

    # 2. Basketball specific heuristics
    clean_sp = sport.lower()
    if clean_sp == "basketball":
        if any(q in code or q in long_desc for q in ("Q1", "1ST")):
            return MatchStatus.LIVE, "Q1", True
        if any(q in code or q in long_desc for q in ("Q2", "2ND")):
            return MatchStatus.LIVE, "Q2", True
        if any(q in code or q in long_desc for q in ("Q3", "3RD")):
            return MatchStatus.LIVE, "Q3", True
        if any(q in code or q in long_desc for q in ("Q4", "4TH")):
            return MatchStatus.LIVE, "Q4", True
        if "OT" in code or "OVERTIME" in long_desc:
            return MatchStatus.LIVE, "OT", True

    # 3. Tennis specific heuristics
    if clean_sp == "tennis":
        if any(s in code or s in long_desc for s in ("S1", "SET 1", "1ST SET")):
            return MatchStatus.LIVE, "S1", True
        if any(s in code or s in long_desc for s in ("S2", "SET 2", "2ND SET")):
            return MatchStatus.LIVE, "S2", True
        if any(s in code or s in long_desc for s in ("S3", "SET 3", "3RD SET")):
            return MatchStatus.LIVE, "S3", True
        if any(s in code or s in long_desc for s in ("S4", "SET 4", "4TH SET")):
            return MatchStatus.LIVE, "S4", True
        if any(s in code or s in long_desc for s in ("S5", "SET 5", "5TH SET")):
            return MatchStatus.LIVE, "S5", True

    # 4. Football specific heuristics
    if clean_sp in ("football", "soccer"):
        if "1H" in code or "FIRST HALF" in long_desc:
            return MatchStatus.LIVE, f"{elapsed}'" if elapsed else "1H", True
        if "2H" in code or "SECOND HALF" in long_desc:
            return MatchStatus.LIVE, f"{elapsed}'" if elapsed else "2H", True
        if "HT" in code or "HALF" in long_desc:
            return MatchStatus.LIVE, "HT", True

    # Default fallback: Scheduled
    return MatchStatus.SCHEDULED, "NS", False


def _parse_utc_datetime(val: Any) -> datetime:
    """Parse timestamp, ISO string, or fallback to now + 3 hours."""
    if isinstance(val, datetime):
        return val if val.tzinfo is not None else val.replace(tzinfo=timezone.utc)
    if isinstance(val, (int, float)):
        return datetime.fromtimestamp(val, tz=timezone.utc)
    if isinstance(val, str) and val.strip():
        try:
            dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass
    return datetime.now(timezone.utc) + timedelta(hours=3)


# --------------------------------------------------------------------------- #
# Payloads Normalization
# --------------------------------------------------------------------------- #
def normalize_api_sports_payload(
    sport: str, raw_item: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Transform raw API-Sports item from any sport into standard match dictionary.

    Args:
        sport: 'football', 'basketball', or 'tennis'.
        raw_item: Raw dictionary from API-Sports 'response' array.

    Returns:
        Standardized fixture dictionary or None if invalid.
    """
    clean_sp = sport.strip().lower()
    if clean_sp == "soccer":
        clean_sp = "football"

    home_name = ""
    away_name = ""
    home_logo: Optional[str] = None
    away_logo: Optional[str] = None
    comp_name = ""
    kickoff_raw: Any = None
    raw_status_code = ""
    raw_long_status = ""
    elapsed: Optional[Any] = None
    home_score: Optional[int] = None
    away_score: Optional[int] = None

    if clean_sp == "football":
        fixture_data = raw_item.get("fixture") or {}
        league_data = raw_item.get("league") or {}
        teams_data = raw_item.get("teams") or {}
        goals_data = raw_item.get("goals") or {}
        status_data = fixture_data.get("status") or {}

        home_obj = teams_data.get("home") or {}
        away_obj = teams_data.get("away") or {}
        home_name = home_obj.get("name") or ""
        away_name = away_obj.get("name") or ""
        home_logo = home_obj.get("logo")
        away_logo = away_obj.get("logo")

        comp_name = league_data.get("name") or "Premier League"
        if league_data.get("country"):
            comp_name = f"{league_data['country']} - {comp_name}"

        kickoff_raw = fixture_data.get("date") or fixture_data.get("timestamp")
        raw_status_code = status_data.get("short") or ""
        raw_long_status = status_data.get("long") or ""
        elapsed = status_data.get("elapsed")

        home_score = goals_data.get("home")
        away_score = goals_data.get("away")

    elif clean_sp == "basketball":
        league_data = raw_item.get("league") or {}
        teams_data = raw_item.get("teams") or {}
        scores_data = raw_item.get("scores") or {}
        status_data = raw_item.get("status") or {}

        home_obj = teams_data.get("home") or {}
        away_obj = teams_data.get("away") or {}
        home_name = home_obj.get("name") or ""
        away_name = away_obj.get("name") or ""
        home_logo = home_obj.get("logo")
        away_logo = away_obj.get("logo")

        comp_name = league_data.get("name") or "NBA"

        kickoff_raw = raw_item.get("date") or raw_item.get("timestamp")
        raw_status_code = status_data.get("short") or ""
        raw_long_status = status_data.get("long") or ""
        elapsed = status_data.get("timer")

        h_score_obj = scores_data.get("home")
        a_score_obj = scores_data.get("away")
        if isinstance(h_score_obj, dict):
            home_score = h_score_obj.get("total")
        elif isinstance(h_score_obj, (int, float)):
            home_score = int(h_score_obj)

        if isinstance(a_score_obj, dict):
            away_score = a_score_obj.get("total")
        elif isinstance(a_score_obj, (int, float)):
            away_score = int(a_score_obj)

    elif clean_sp == "tennis":
        tourn_data = raw_item.get("tournament") or raw_item.get("league") or {}
        players_data = raw_item.get("players") or raw_item.get("teams") or {}
        scores_data = raw_item.get("scores") or raw_item.get("score") or {}
        status_data = raw_item.get("status") or {}

        home_obj = players_data.get("home") or {}
        away_obj = players_data.get("away") or {}
        home_name = home_obj.get("name") or ""
        away_name = away_obj.get("name") or ""
        home_logo = home_obj.get("logo")
        away_logo = away_obj.get("logo")

        comp_name = tourn_data.get("name") or "ATP Tour"

        kickoff_raw = raw_item.get("date") or raw_item.get("timestamp")
        raw_status_code = status_data.get("short") or ""
        raw_long_status = status_data.get("long") or ""

        h_score_obj = scores_data.get("home")
        a_score_obj = scores_data.get("away")
        if isinstance(h_score_obj, dict):
            home_score = h_score_obj.get("total")
        elif isinstance(h_score_obj, (int, float)):
            home_score = int(h_score_obj)

        if isinstance(a_score_obj, dict):
            away_score = a_score_obj.get("total")
        elif isinstance(a_score_obj, (int, float)):
            away_score = int(a_score_obj)

    home_name = home_name.strip()
    away_name = away_name.strip()
    if not home_name or not away_name or home_name == away_name:
        return None

    # Map status
    mapped_status, display_period, is_live = standardize_match_status(
        clean_sp, raw_status_code, raw_long_status, elapsed
    )
    kickoff_dt = _parse_utc_datetime(kickoff_raw)
    is_individual = (clean_sp == "tennis")

    # Format score object for UI consumption
    score_obj = None
    if is_live or mapped_status == MatchStatus.FINISHED or (home_score is not None and away_score is not None):
        score_obj = {
            "home": home_score if home_score is not None else 0,
            "away": away_score if away_score is not None else 0,
            "status": "FT" if mapped_status == MatchStatus.FINISHED else ("LIVE" if is_live else "NS"),
            "current_minute": display_period if is_live else ("FT" if mapped_status == MatchStatus.FINISHED else "NS"),
            "period": display_period,
            "is_live": is_live,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    return {
        "sport": clean_sp,
        "competition": comp_name,
        "home_team": home_name,
        "away_team": away_name,
        "home_logo_url": home_logo,
        "away_logo_url": away_logo,
        "kickoff_time": kickoff_dt,
        "status": mapped_status,
        "final_score": score_obj,
        "actual_home_score": home_score if (is_live or mapped_status == MatchStatus.FINISHED) else None,
        "actual_away_score": away_score if (is_live or mapped_status == MatchStatus.FINISHED) else None,
        "is_individual": is_individual,
        "raw_period": display_period,
        "raw_payload": raw_item,
    }


# --------------------------------------------------------------------------- #
# Database Ingestion & Upsert
# --------------------------------------------------------------------------- #
def ingest_fixtures_to_db(
    fixtures: List[Dict[str, Any]], db_session: Session
) -> Dict[str, int]:
    """Upsert normalized fixtures into the universal Match SQLAlchemy schema.

    - Resolves or creates canonical Sport and Entity rows.
    - Matches existing records in the +/- 6 hour window on (sport_id, home_entity_id, away_entity_id).
    - Inserts new fixtures and establishes baseline odds.

    Args:
        fixtures: List of normalized match dicts.
        db_session: Active SQLAlchemy session.

    Returns:
        Summary dict containing counts of inserted, updated, and total records.
    """
    inserted = 0
    updated = 0
    skipped = 0

    if not fixtures:
        return {"inserted": 0, "updated": 0, "skipped": 0, "total": 0}

    for fix in fixtures:
        sport_slug = fix["sport"]
        home_name = fix["home_team"]
        away_name = fix["away_team"]
        comp_name = fix["competition"]
        kickoff_dt = fix["kickoff_time"]
        status = fix["status"]
        final_score = fix.get("final_score")
        actual_h = fix.get("actual_home_score")
        actual_a = fix.get("actual_away_score")
        is_indiv = fix.get("is_individual", False)
        home_logo = fix.get("home_logo_url")
        away_logo = fix.get("away_logo_url")

        # 1. Resolve Sport
        sport = _get_or_create_sport(db_session, sport_slug)

        # 2. Resolve Competitor Entities
        home_entity = _get_or_create_entity(
            db_session, sport.id, home_name, is_individual=is_indiv, logo_url=home_logo
        )
        away_entity = _get_or_create_entity(
            db_session, sport.id, away_name, is_individual=is_indiv, logo_url=away_logo
        )

        if home_entity.id == away_entity.id:
            skipped += 1
            continue

        # 3. Lookup existing match in the +/- 6 hours window
        match_date = kickoff_dt.date()
        start_w = datetime.combine(match_date, datetime.min.time(), tzinfo=timezone.utc) - timedelta(hours=6)
        end_w = datetime.combine(match_date, datetime.max.time(), tzinfo=timezone.utc) + timedelta(hours=6)

        existing_match = db_session.scalar(
            select(Match).where(
                Match.sport_id == sport.id,
                Match.home_entity_id == home_entity.id,
                Match.away_entity_id == away_entity.id,
                Match.kickoff_time >= start_w,
                Match.kickoff_time <= end_w,
            )
        )

        if existing_match:
            existing_match.status = status
            existing_match.kickoff_time = kickoff_dt
            if comp_name and not existing_match.competition_name:
                existing_match.competition_name = comp_name
            if final_score:
                existing_match.final_score = final_score
            if actual_h is not None:
                existing_match.actual_home_score = actual_h
            if actual_a is not None:
                existing_match.actual_away_score = actual_a
            updated += 1
        else:
            new_match = Match(
                sport_id=sport.id,
                competition_name=comp_name,
                home_entity_id=home_entity.id,
                away_entity_id=away_entity.id,
                kickoff_time=kickoff_dt,
                status=status,
                final_score=final_score,
                actual_home_score=actual_h,
                actual_away_score=actual_a,
            )
            db_session.add(new_match)
            db_session.flush()

            # Insert baseline bookmaker odds
            try:
                _insert_baseline_odds(db_session, new_match, fix, sport_slug)
            except Exception as b_err:
                log.debug("Could not insert baseline odds for match %s: %s", new_match.id, b_err)

            inserted += 1

    try:
        db_session.commit()
    except Exception as exc:
        db_session.rollback()
        log.error("Failed to commit multi-sport ingestion batch: %s", exc)
        raise

    return {
        "inserted": inserted,
        "updated": updated,
        "skipped": skipped,
        "total": len(fixtures),
    }


# --------------------------------------------------------------------------- #
# End-to-End Multi-Sport Pipeline
# --------------------------------------------------------------------------- #
async def run_multi_sport_pipeline(
    date_str: Optional[str] = None,
    sports: Optional[List[str]] = None,
    session: Optional[Session] = None,
    client: Optional[APISportsClient] = None,
    live: bool = False,
) -> Dict[str, Any]:
    """Execute complete multi-sport data ingestion pipeline.

    1. Fetches fixtures across Football, Basketball, and Tennis endpoints.
    2. Normalizes JSON payloads and standardizes status codes (Q1, S1, FT, etc.).
    3. Ingests into the local database schema with zero schema errors.

    Args:
        date_str: Target date in 'YYYY-MM-DD' format (defaults to today UTC).
        sports: List of sports to sync (defaults to ['football', 'basketball', 'tennis']).
        session: Active SQLAlchemy session. If None, creates a new session.
        client: APISportsClient instance. If None, uses singleton.
        live: If True, fetches live matches.

    Returns:
        Structured pipeline execution summary.
    """
    target_date = date_str or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    target_sports = sports or ["football", "basketball", "tennis"]
    http_client = client or api_sports_client

    results_by_sport: Dict[str, Any] = {}
    all_normalized: List[Dict[str, Any]] = []

    log.info(
        "Initiating Multi-Sport Pipeline on %s (Live=%s, Sports=%s)...",
        target_date,
        live,
        target_sports,
    )

    for sp in target_sports:
        try:
            raw_fixtures = await http_client.fetch_fixtures(
                sport=sp, date_str=target_date, live=live
            )
            normalized_list = []
            for raw in raw_fixtures:
                norm = normalize_api_sports_payload(sp, raw)
                if norm:
                    normalized_list.append(norm)

            all_normalized.extend(normalized_list)
            results_by_sport[sp] = {
                "raw_fetched": len(raw_fixtures),
                "normalized": len(normalized_list),
            }
        except Exception as err:
            log.error("Pipeline failure fetching sport %s: %s", sp, err)
            results_by_sport[sp] = {"error": str(err), "raw_fetched": 0, "normalized": 0}

    # Ingest all normalized fixtures into the database
    def _execute_ingest(s: Session) -> Dict[str, int]:
        return ingest_fixtures_to_db(all_normalized, s)

    if session is not None:
        ingest_stats = _execute_ingest(session)
    else:
        with SessionLocal() as s:
            ingest_stats = _execute_ingest(s)

    summary = {
        "status": "success",
        "date": target_date,
        "is_live": live,
        "sports_summary": results_by_sport,
        "total_normalized": len(all_normalized),
        "db_stats": ingest_stats,
    }

    log.info("Multi-Sport Pipeline completed successfully: %s", summary)
    return summary
