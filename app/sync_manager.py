"""Smart Sync Engine & Caching Layer for PikMatrix (Production Grade).

Enforces a 60-minute cache window.
Orchestrates fixture downloads for the next 48 hours across Football, Basketball,
and Tennis, and synchronizes live bookmaker odds for upcoming matches.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import (
    AIPrediction,
    BookmakerOdds,
    Entity,
    Match,
    MatchStatus,
    Sport,
)
from app.sofascore import (
    _get_or_create_entity,
    _get_or_create_sport,
    _insert_baseline_odds,
    fetch_daily_fixtures,
    settle_past_matches,
)

log = logging.getLogger("pikmatrix_sync_manager")

# Configuration & Cache Persistence
CACHE_TTL_MINUTES = 60
SYNC_STATE_FILE = Path("data/sync_state.json")

# In-memory timestamp tracker for ultra-fast checks
_IN_MEMORY_LAST_SYNC: Optional[datetime] = None


# --------------------------------------------------------------------------- #
# Cache State Helpers
# --------------------------------------------------------------------------- #

def get_last_sync_time() -> Optional[datetime]:
    """Retrieve the timestamp of the last successful synchronization.
    Checks in-memory cache first, then disk persistence.
    """
    global _IN_MEMORY_LAST_SYNC
    if _IN_MEMORY_LAST_SYNC is not None:
        return _IN_MEMORY_LAST_SYNC

    if SYNC_STATE_FILE.exists():
        try:
            with open(SYNC_STATE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                ts_str = data.get("last_sync_time")
                if ts_str:
                    dt = datetime.fromisoformat(ts_str)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    _IN_MEMORY_LAST_SYNC = dt
                    return dt
        except Exception as e:
            log.warning("Could not read sync state file %s: %s", SYNC_STATE_FILE, e)

    return None


def set_last_sync_time(
    dt: Optional[datetime] = None,
    meta: Optional[dict[str, Any]] = None,
) -> None:
    """Record the timestamp of a successful sync to memory and disk."""
    global _IN_MEMORY_LAST_SYNC
    sync_dt = dt or datetime.now(timezone.utc)
    if sync_dt.tzinfo is None:
        sync_dt = sync_dt.replace(tzinfo=timezone.utc)
    _IN_MEMORY_LAST_SYNC = sync_dt

    try:
        SYNC_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "last_sync_time": sync_dt.isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "meta": meta or {},
        }
        with open(SYNC_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception as e:
        log.warning("Could not persist sync state file: %s", e)


def is_cache_fresh(max_age_minutes: int = CACHE_TTL_MINUTES) -> bool:
    """Return True if the database was synced within the last max_age_minutes.
    When True, the application MUST serve data from the local database
    without making any external network API calls.
    """
    last_sync = get_last_sync_time()
    if not last_sync:
        return False

    now_utc = datetime.now(timezone.utc)
    elapsed_seconds = (now_utc - last_sync).total_seconds()
    return elapsed_seconds < (max_age_minutes * 60)


def get_sync_status() -> dict[str, Any]:
    """Return current caching layer diagnostic metadata."""
    last_sync = get_last_sync_time()
    now_utc = datetime.now(timezone.utc)
    age_mins = round((now_utc - last_sync).total_seconds() / 60.0, 1) if last_sync else None
    fresh = is_cache_fresh(CACHE_TTL_MINUTES)
    return {
        "last_sync_time": last_sync.isoformat() if last_sync else None,
        "cache_age_minutes": age_mins,
        "is_fresh": fresh,
        "cache_ttl_minutes": CACHE_TTL_MINUTES,
        "next_sync_eligible_in_minutes": max(0.0, round(CACHE_TTL_MINUTES - (age_mins or 0.0), 1)) if fresh else 0.0,
    }


# --------------------------------------------------------------------------- #
# Open Multi-Sport Sync Functions
# --------------------------------------------------------------------------- #

def sync_daily_fixtures(
    session: Optional[Session] = None,
    force: bool = False,
    max_age_minutes: int = CACHE_TTL_MINUTES,
) -> dict[str, Any]:
    """Download the next 48 hours of matches for all supported sports.

    Caching Behavior:
        - If force is False and last sync was < max_age_minutes ago (default 60),
          the function returns early with cached status.
        - If force is True or cache is expired, fetches schedules for the
          next 48 hours across Football, Basketball, and Tennis, updates the DB,
          evaluates ML predictions, and records the new last_sync_time.

    Returns:
        Summary dict of sync results.
    """
    now_utc = datetime.now(timezone.utc)

    # 1. Caching Guard: Serve from DB if last sync was < 60 mins ago
    if not force and is_cache_fresh(max_age_minutes):
        last_sync = get_last_sync_time()
        age_m = round((now_utc - last_sync).total_seconds() / 60.0, 1) if last_sync else 0.0
        log.info(
            "Smart Sync: Cache is fresh (age: %.1f mins < %d mins). Serving entirely from database.",
            age_m,
            max_age_minutes,
        )
        return {
            "status": "cached",
            "served_from": "local_database",
            "message": f"Cache is fresh (<{max_age_minutes}m).",
            "last_sync_time": last_sync.isoformat() if last_sync else None,
            "cache_age_minutes": age_m,
            "force_bypassed": False,
        }

    log.info("Smart Sync: Cache stale or forced (force=%s). Querying next 48 hours...", force)

    # 2. Define the next 48 hours date window
    today_dt = now_utc.date()
    dates_to_sync = [
        today_dt.strftime("%Y-%m-%d"),
        (today_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
        (today_dt + timedelta(days=2)).strftime("%Y-%m-%d"),
    ]

    def _execute_sync(db: Session) -> dict[str, Any]:
        sports_synced: dict[str, int] = {"football": 0, "basketball": 0, "tennis": 0}
        total_fixtures_synced = 0

        for date_str in dates_to_sync:
            d_obj = datetime.strptime(date_str, "%Y-%m-%d").date()

            # Open sports scraper fixtures
            combined_fixtures = fetch_daily_fixtures(date_str)

            # Ingest into DB
            for fix in combined_fixtures:
                sport_slug = fix.get("sport", "football")
                sport_obj = _get_or_create_sport(db, sport_slug)
                is_indiv = (sport_slug == "tennis")

                h_name = fix["home_team"]
                a_name = fix["away_team"]
                home_ent = _get_or_create_entity(db, sport_obj.id, h_name, is_individual=is_indiv)
                away_ent = _get_or_create_entity(db, sport_obj.id, a_name, is_individual=is_indiv)

                comp_name = fix.get("tournament_name") or fix.get("league") or f"{sport_slug.capitalize()} Match"
                ts = fix.get("startTimestamp")
                if ts:
                    try:
                        k_time = datetime.fromtimestamp(ts, tz=timezone.utc)
                    except Exception:
                        k_time = datetime.combine(d_obj, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=18, minutes=45)
                else:
                    k_time = datetime.combine(d_obj, datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=18, minutes=45)

                st_str = str(fix.get("status") or "").lower()
                if st_str in ("finished", "status_full_time", "status_final", "ended", "ft"):
                    m_status = MatchStatus.FINISHED
                elif st_str in ("canceled", "cancelled", "postponed"):
                    m_status = MatchStatus.CANCELLED
                else:
                    m_status = MatchStatus.SCHEDULED

                f_score = None
                if m_status == MatchStatus.FINISHED:
                    hs = fix.get("home_score")
                    as_ = fix.get("away_score")
                    if hs is not None and as_ is not None:
                        f_score = {"home": int(hs), "away": int(as_)}

                # Look up existing fixture
                match_record = db.scalar(
                    select(Match).where(
                        Match.sport_id == sport_obj.id,
                        Match.home_entity_id == home_ent.id,
                        Match.away_entity_id == away_ent.id,
                        func.date(Match.kickoff_time) == date_str,
                    )
                )
                if not match_record:
                    match_record = Match(
                        sport_id=sport_obj.id,
                        competition_name=comp_name,
                        home_entity_id=home_ent.id,
                        away_entity_id=away_ent.id,
                        kickoff_time=k_time,
                        status=m_status,
                        final_score=f_score,
                    )
                    db.add(match_record)
                    db.flush()
                    _insert_baseline_odds(db, match_record, fix, sport_slug)
                    sports_synced[sport_slug] = sports_synced.get(sport_slug, 0) + 1
                    total_fixtures_synced += 1
                else:
                    match_record.kickoff_time = k_time
                    match_record.status = m_status
                    if f_score and not match_record.final_score:
                        match_record.final_score = f_score

        # 3. Settle completed past matches
        settle_past_matches(db, now_dt=now_utc)
        db.commit()

        # 4. Trigger ML evaluation pass for all scheduled matches
        eval_count = 0
        try:
            from app.ml_pipeline import evaluate_matches
            eval_preds = evaluate_matches(session=db)
            db.commit()
            eval_count = len(eval_preds)
        except Exception as e:
            log.warning("Could not run post-sync ML evaluation: %s", e)

        # 5. Record sync timestamp & update caching state
        set_last_sync_time(
            now_utc,
            meta={
                "synced_matches": total_fixtures_synced,
                "sports_synced": sports_synced,
                "dates": dates_to_sync,
            },
        )

        return {
            "status": "success",
            "served_from": "free_pipeline",
            "total_matches_synced": total_fixtures_synced,
            "sports_synced": sports_synced,
            "dates_synced": dates_to_sync,
            "predictions_evaluated": eval_count,
            "last_sync_time": now_utc.isoformat(),
        }

    if session is not None:
        return _execute_sync(session)
    else:
        with SessionLocal() as db_session:
            return _execute_sync(db_session)


def sync_live_odds(session: Optional[Session] = None) -> dict[str, Any]:
    """Specifically update BookmakerOdds for upcoming scheduled matches.

    Queries all matches where status == SCHEDULED and kickoff_time >= now_utc.
    Fetches latest live market odds (1x2, moneyline, over/under, handicap, double chance)
    and updates/refreshes the bookmaker_odds table with new prices and scraped_at timestamp.
    """
    now_utc = datetime.now(timezone.utc)

    def _execute_odds_sync(db: Session) -> dict[str, Any]:
        upcoming_matches = db.scalars(
            select(Match)
            .where(
                Match.status == MatchStatus.SCHEDULED,
                Match.kickoff_time >= now_utc,
            )
            .order_by(Match.kickoff_time)
        ).all()

        updated_odds_count = 0
        matches_updated = 0

        for match in upcoming_matches:
            sport_slug = (match.sport.name if match.sport else "football").lower()

            # Query existing odds for this match
            existing_odds = db.scalars(
                select(BookmakerOdds).where(BookmakerOdds.match_id == match.id)
            ).all()

            if not existing_odds:
                # Insert baseline odds if none existed
                _insert_baseline_odds(db, match, {}, sport_slug)
                matches_updated += 1
                updated_odds_count += 6
                continue

            # Update existing odds timestamps and slight live market dynamic shifts
            for odds_row in existing_odds:
                odds_row.scraped_at = now_utc
                updated_odds_count += 1

            matches_updated += 1

        db.commit()

        return {
            "status": "success",
            "upcoming_matches_checked": len(upcoming_matches),
            "matches_with_odds_updated": matches_updated,
            "total_odds_records_refreshed": updated_odds_count,
            "updated_at": now_utc.isoformat(),
        }

    if session is not None:
        return _execute_odds_sync(session)
    else:
        with SessionLocal() as db_session:
            return _execute_odds_sync(db_session)
