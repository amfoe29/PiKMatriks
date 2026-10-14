"""REST API and Automation Pipeline Router (Phase 2 Restructuring).

Serves programmatic JSON endpoints, sync sweeps, settlement triggers, and telemetry.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from app.database import get_db
from app.models import AIPrediction, Match, MatchStatus, Sport, User
from app.ml_pipeline import FeatureBuilder, generate_two_tier_predictions
from app.scraper import fetch_deep_match_stats, fetch_player_metrics
from app.services.auth_service import get_current_user
from app.services.match_service import enrich_match_for_ui, get_model_retrain_metadata
from app.settlement_engine import run_settlement_sweep
from app.sofascore import (
    settle_past_matches,
    sync_all_dates,
    sync_daily_fixtures,
)
from app.sync_engine import (
    enrich_matches_with_telemetry,
    grade_finished_predictions,
    sync_daily_schedule,
)
from app.sync_manager import (
    get_sync_status,
    sync_daily_fixtures as smart_sync_fixtures,
    sync_live_odds as smart_sync_live_odds,
)

log = logging.getLogger("pikmatrix_router_api")
router = APIRouter(tags=["REST API & Automation"])


@router.get("/health", response_class=JSONResponse)
def health_check(db: Session = Depends(get_db)):
    """Health check endpoint validating database connectivity and prediction counts."""
    count = db.scalar(select(func.count(AIPrediction.id)))
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_predictions_stored": count,
    }


@router.get("/api/retrain-status", response_class=JSONResponse)
def get_retrain_status_api():
    """Discreet internal endpoint returning model retraining and learning telemetry."""
    return JSONResponse(content=get_model_retrain_metadata())


@router.get("/api/matches/feed", response_class=JSONResponse)
def get_matches_feed_api(
    sport: Optional[str] = None,
    safe_only: bool = False,
    date: Optional[str] = None,
    filter: Optional[str] = None,
    date_filter: Optional[str] = None,
    competition: Optional[str] = None,
    q: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """JSON API endpoint returning match feed with sport counts and predictions."""
    now_utc = datetime.now(timezone.utc)
    current_date = now_utc.date()
    yesterday_date = current_date - timedelta(days=1)
    tomorrow_date = current_date + timedelta(days=1)

    active_date_filter = (filter or date_filter or "").lower()
    if date:
        date_clean = date.strip()[:10]
        if date_clean == current_date.strftime("%Y-%m-%d"):
            active_date_filter = "today"
        elif date_clean == yesterday_date.strftime("%Y-%m-%d"):
            active_date_filter = "yesterday"
        elif date_clean == tomorrow_date.strftime("%Y-%m-%d"):
            active_date_filter = "tomorrow"
        else:
            active_date_filter = date_clean
    elif not active_date_filter:
        active_date_filter = "today"

    match_query = (
        select(Match)
        .options(
            joinedload(Match.sport),
            joinedload(Match.home_entity),
            joinedload(Match.away_entity),
        )
        .order_by(Match.kickoff_time)
    )

    if active_date_filter == "today":
        start_dt = datetime.combine(current_date, datetime.min.time())
        end_dt = start_dt + timedelta(days=1)
        match_query = match_query.where(Match.kickoff_time >= start_dt, Match.kickoff_time < end_dt)
    elif active_date_filter == "yesterday":
        start_dt = datetime.combine(yesterday_date, datetime.min.time())
        end_dt = start_dt + timedelta(days=1)
        match_query = match_query.where(Match.kickoff_time >= start_dt, Match.kickoff_time < end_dt)
    elif active_date_filter == "tomorrow":
        start_dt = datetime.combine(tomorrow_date, datetime.min.time())
        end_dt = start_dt + timedelta(days=1)
        match_query = match_query.where(Match.kickoff_time >= start_dt, Match.kickoff_time < end_dt)
    elif active_date_filter == "upcoming":
        match_query = match_query.where(Match.kickoff_time >= now_utc.replace(tzinfo=None))
    elif active_date_filter and active_date_filter != "all":
        try:
            custom_d = datetime.strptime(active_date_filter, "%Y-%m-%d").date()
            start_dt = datetime.combine(custom_d, datetime.min.time())
            end_dt = start_dt + timedelta(days=1)
            match_query = match_query.where(Match.kickoff_time >= start_dt, Match.kickoff_time < end_dt)
        except Exception:
            pass

    scheduled_matches = db.scalars(match_query).all()

    deduped_matches = []
    seen_fixtures = set()
    for m in scheduled_matches:
        home_name = (m.home_entity.name if m.home_entity else str(m.home_entity_id or "")).strip().lower()
        away_name = (m.away_entity.name if m.away_entity else str(m.away_entity_id or "")).strip().lower()
        match_date = m.kickoff_time.date() if m.kickoff_time else None
        fixture_key = (m.sport_id, home_name, away_name, match_date)
        if fixture_key in seen_fixtures:
            continue
        seen_fixtures.add(fixture_key)
        deduped_matches.append(m)

    feature_builder = FeatureBuilder()
    all_feed_matches = []
    sport_match_counts = {"football": 0, "basketball": 0, "tennis": 0}

    for m in deduped_matches:
        sp_name = (m.sport.name if m.sport else "football").lower()
        sp_key = "basketball" if "basket" in sp_name or "nba" in sp_name else ("tennis" if "ten" in sp_name else "football")
        if sp_key in sport_match_counts:
            sport_match_counts[sp_key] += 1

        if sport and sport.lower() != "all" and sp_key != sport.lower():
            continue
        if competition and (not m.competition_name or competition.lower() not in m.competition_name.lower()):
            continue
        if q:
            q_clean = q.strip().lower()
            h_n = (m.home_entity.name if m.home_entity else "").lower()
            a_n = (m.away_entity.name if m.away_entity else "").lower()
            c_n = (m.competition_name or "").lower()
            if q_clean not in h_n and q_clean not in a_n and q_clean not in c_n:
                continue

        m_dict, _ = enrich_match_for_ui(m, db, feature_builder)
        if safe_only and not m_dict.get("has_safe_pick"):
            continue
        all_feed_matches.append(m_dict)

    sport_match_counts["all"] = len(deduped_matches)
    return {
        "status": "success",
        "date_filter": active_date_filter,
        "total_matches": len(all_feed_matches),
        "sport_counts": sport_match_counts,
        "matches": all_feed_matches,
    }


@router.get("/api/predictions", response_class=JSONResponse)
def get_predictions_api(
    sport: Optional[str] = None,
    safe_only: bool = False,
    date: Optional[str] = None,
    filter: Optional[str] = None,
    date_filter: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """JSON API endpoint returning active predictions for programmatic consumption."""
    now_utc = datetime.now(timezone.utc)
    current_date = now_utc.date()
    yesterday_date = current_date - timedelta(days=1)
    tomorrow_date = current_date + timedelta(days=1)

    active_date_filter = (filter or date_filter or "").lower()
    if date:
        date_clean = date.strip()[:10]
        if date_clean == current_date.strftime("%Y-%m-%d"):
            active_date_filter = "today"
        elif date_clean == yesterday_date.strftime("%Y-%m-%d"):
            active_date_filter = "yesterday"
        elif date_clean == tomorrow_date.strftime("%Y-%m-%d"):
            active_date_filter = "tomorrow"
        else:
            active_date_filter = date_clean

    match_query = (
        select(Match)
        .options(
            joinedload(Match.sport),
            joinedload(Match.home_entity),
            joinedload(Match.away_entity),
        )
        .order_by(Match.kickoff_time)
    )

    if active_date_filter == "today":
        start_dt = datetime.combine(current_date, datetime.min.time())
        end_dt = start_dt + timedelta(days=1)
        match_query = match_query.where(Match.kickoff_time >= start_dt, Match.kickoff_time < end_dt)
    elif active_date_filter == "yesterday":
        start_dt = datetime.combine(yesterday_date, datetime.min.time())
        end_dt = start_dt + timedelta(days=1)
        match_query = match_query.where(Match.kickoff_time >= start_dt, Match.kickoff_time < end_dt)
    elif active_date_filter == "tomorrow":
        start_dt = datetime.combine(tomorrow_date, datetime.min.time())
        end_dt = start_dt + timedelta(days=1)
        match_query = match_query.where(Match.kickoff_time >= start_dt, Match.kickoff_time < end_dt)
    elif active_date_filter == "upcoming":
        match_query = match_query.where(Match.kickoff_time >= now_utc.replace(tzinfo=None))
    elif active_date_filter and active_date_filter != "all":
        try:
            custom_d = datetime.strptime(active_date_filter, "%Y-%m-%d").date()
            start_dt = datetime.combine(custom_d, datetime.min.time())
            end_dt = start_dt + timedelta(days=1)
            match_query = match_query.where(Match.kickoff_time >= start_dt, Match.kickoff_time < end_dt)
        except Exception:
            pass

    matches = db.scalars(match_query).all()
    results = []
    feature_builder = FeatureBuilder()

    for m in matches:
        sp_name = (m.sport.name if m.sport else "football").lower()
        if sport and sport.lower() not in sp_name:
            continue
        h_name = m.home_entity.name if m.home_entity else f"Team_{m.home_entity_id}"
        a_name = m.away_entity.name if m.away_entity else f"Team_{m.away_entity_id}"
        is_fin = (m.status == MatchStatus.FINISHED)

        two_tier = generate_two_tier_predictions(m, session=db, builder=feature_builder)
        active_plays = two_tier.get("active_plays", [])
        arch = two_tier.get("archetype", {})

        db_preds = db.scalars(select(AIPrediction).where(AIPrediction.match_id == m.id)).all()
        db_model_map = {p.model_name: p.id for p in db_preds}
        db_pred_map = {p.market_type.lower(): p.id for p in db_preds}

        for p in active_plays:
            pred_id = db_model_map.get(p.get("model_name"), db_pred_map.get(p["market_type"].lower(), p.get("id")))

            results.append({
                "id": pred_id,
                "match_id": m.id,
                "sport": m.sport.name.capitalize() if m.sport else "Football",
                "competition": m.competition_name,
                "home_team": h_name,
                "away_team": a_name,
                "fixture": f"{h_name} vs {a_name}",
                "kickoff_time": m.kickoff_time.isoformat() if m.kickoff_time else None,
                "status": m.status.value if hasattr(m.status, "value") else str(m.status),
                "is_finished": is_fin,
                "final_score": m.final_score if is_fin else None,
                "tier": p.get("tier"),
                "tier_title": p.get("tier_title"),
                "market_type": p["market_type"],
                "selection": p["selection"],
                "confidence_score": p["confidence_score"],
                "confidence_num": p.get("confidence_num"),
                "decimal_odds": round(float(p["decimal_odds"]), 2) if (p.get("decimal_odds") is not None and not isinstance(p.get("decimal_odds"), dict)) else ((p.get("decimal_odds", {}).get("decimal_odds") or p.get("decimal_odds", {}).get("odds")) if isinstance(p.get("decimal_odds"), dict) else None),
                "implied_prob": p.get("implied_prob"),
                "alpha_edge": p.get("alpha_edge"),
                "projected_stat": p["projected_stat"],
                "insight": p["insight"],
                "why_this_pick": p.get("why_this_pick"),
                "archetype": arch,
                "line": p.get("line"),
                "is_safe_pick": True,
                "settlement_status": p.get("settlement_status") or (p.get("settlement_badge", {}).get("status") if isinstance(p.get("settlement_badge"), dict) else None),
                "settlement_detail": p.get("settlement_detail") or (p.get("settlement_badge", {}).get("detail") if isinstance(p.get("settlement_badge"), dict) else None),
                "settlement_badge": (p.get("settlement_badge", {}).get("badge") if isinstance(p.get("settlement_badge"), dict) else p.get("settlement_badge")),
            })

    return {"count": len(results), "predictions": results}


@router.post("/api/sync", response_class=JSONResponse)
async def api_sync_fixtures(
    request: Request,
    date: Optional[str] = Query(None, description="Target date in YYYY-MM-DD"),
    purge_all: bool = Query(False, description="Purge old seed matches before syncing"),
    all_dates: bool = Query(False, description="Sync yesterday, today, and tomorrow"),
    db: Session = Depends(get_db),
):
    """Synchronize real-world daily match fixtures from Sofascore into pikmatrix.db."""
    body = {}
    try:
        body = await request.json()
    except Exception:
        pass

    target_date = body.get("date") or date
    do_purge = bool(body.get("purge_all", purge_all))
    do_all_dates = bool(body.get("all_dates", all_dates))

    if do_all_dates or (not target_date and do_purge):
        result = sync_all_dates(session=db, purge_first=do_purge)
    else:
        result = sync_daily_fixtures(
            target_date=target_date,
            session=db,
            purge_all=do_purge,
        )

    return result


@router.post("/api/admin/settle", response_class=JSONResponse)
def api_admin_settle(
    request: Request,
    admin_key: Optional[str] = Query(None, description="Admin secret key for automated workers"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Internal trigger endpoint to execute a deterministic settlement sweep."""
    configured_key = os.getenv("ADMIN_API_KEY", "pikmatrix-admin-2026")
    provided_key = request.headers.get("X-Admin-Key") or admin_key

    if not (current_user or (provided_key and provided_key == configured_key)):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Valid authentication session or X-Admin-Key required to trigger settlement sweep",
        )

    now_utc = datetime.now(timezone.utc)
    try:
        settle_past_matches(db, now_dt=now_utc)
    except Exception as e:
        log.warning("settle_past_matches error before admin sweep: %s", e)

    result = run_settlement_sweep(db)
    return result


@router.post("/api/admin/sync", response_class=JSONResponse)
def api_admin_sync(
    request: Request,
    force: bool = Query(True, description="Force sync ignoring 60-minute cache window"),
    sync_odds: bool = Query(True, description="Also update BookmakerOdds for upcoming matches"),
    admin_key: Optional[str] = Query(None, description="Admin secret key for automated workers"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user),
):
    """Admin endpoint to manually trigger the Smart Sync Engine."""
    configured_key = os.getenv("ADMIN_API_KEY", "pikmatrix-admin-2026")
    provided_key = request.headers.get("X-Admin-Key") or admin_key

    if not (current_user or (provided_key and provided_key == configured_key)):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Valid authentication session or X-Admin-Key required to trigger sync",
        )

    sync_result = smart_sync_fixtures(session=db, force=force)

    odds_result: dict[str, Any] = {}
    if sync_odds:
        try:
            odds_result = smart_sync_live_odds(session=db)
        except Exception as e:
            log.warning("Could not sync live odds: %s", e)
            odds_result = {"status": "error", "message": str(e)}

    settle_res = {}
    try:
        settle_past_matches(db)
        settle_res = run_settlement_sweep(db)
    except Exception as e:
        log.warning("Post-sync settlement error: %s", e)

    return {
        "status": "success",
        "action": "admin_sync",
        "force_applied": force,
        "sync_fixtures": sync_result,
        "sync_odds": odds_result,
        "settlement_sweep": settle_res,
        "cache_status": get_sync_status(),
    }


@router.post("/api/pipeline/grade-predictions")
async def trigger_grade_predictions(
    auto_finish_elapsed: bool = Query(True, description="Auto-resolve concluded fixtures"),
    db: Session = Depends(get_db),
):
    """Trigger automated prediction grading sweep across finished fixtures."""
    try:
        summary = await grade_finished_predictions(db, auto_finish_elapsed=auto_finish_elapsed)
        return {
            "status": "success",
            "message": "Automated prediction grading sweep completed.",
            "data": summary,
        }
    except Exception as exc:
        log.error("Manual grading sweep error: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(exc)},
        )


@router.post("/api/pipeline/enrich-telemetry")
async def trigger_telemetry_enrichment(
    db: Session = Depends(get_db),
    max_matches: Optional[int] = Query(None, description="Max matches to enrich"),
):
    """Trigger deep match telemetry enrichment loop across upcoming matches."""
    res = await enrich_matches_with_telemetry(db, max_matches=max_matches)
    return {"status": "success", "result": res}


@router.get("/api/telemetry/match/{match_id}")
async def get_match_telemetry(match_id: int):
    """Fetch deep match statistics & momentum telemetry for a specific match."""
    data = await fetch_deep_match_stats(match_id)
    return data


@router.get("/api/telemetry/player/{player_name}")
async def get_player_telemetry(player_name: str):
    """Fetch player ratings and historical performance."""
    data = await fetch_player_metrics(player_name)
    return data


@router.post("/api/pipeline/sync-daily-schedule")
async def trigger_sync_daily_schedule(
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format (defaults to today)"),
    db: Session = Depends(get_db),
):
    """Trigger the Universal Daily Harvester to sync football, basketball, and tennis schedules."""
    target_date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    result = await sync_daily_schedule(db, target_date)
    return {"status": "success", "result": result}
