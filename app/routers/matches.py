"""Matches and Terminal View Router (Phase 2 Restructuring).

Serves the main matches hub, detailed match terminals, search, and live auto-poll partials.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, aliased, joinedload

from app.database import get_db
from app.models import Entity, Match, MatchStatus, Sport, User
from app.ml_pipeline import FeatureBuilder, compute_bivariate_poisson_engine
from app.services.auth_service import get_current_user
from app.services.match_service import (
    build_calendar_days,
    enrich_match_for_ui,
    get_model_retrain_metadata,
)
from app.sync_manager import get_sync_status
from app.telemetry_synthesizer import synthesize_entity_telemetry

log = logging.getLogger("pikmatrix_router_matches")
router = APIRouter(tags=["Matches & Live Inference"])


def init_matches_router(templates: Jinja2Templates) -> APIRouter:
    """Factory to instantiate matches router with Jinja2Templates reference."""

    @router.get("/", response_class=HTMLResponse)
    @router.get("/matches", response_class=HTMLResponse)
    def dashboard(
        request: Request,
        sport: Optional[str] = Query(None, description="Filter by sport: football, basketball, tennis"),
        safe_only: bool = Query(False, description="Filter for safe picks only"),
        q: Optional[str] = Query(None, description="Search term for team or competition"),
        date: Optional[str] = Query(None, description="Filter by exact date YYYY-MM-DD"),
        filter: Optional[str] = Query(None, description="Filter by date keyword: today, yesterday, tomorrow, upcoming, all"),
        date_filter: Optional[str] = Query(None, description="Alias for filter"),
        db: Session = Depends(get_db),
        current_user: Optional[User] = Depends(get_current_user),
    ):
        """Main dashboard view rendering live AI predictions grouped by match in a Sofascore-style layout."""
        if not current_user:
            if request.headers.get("HX-Request") == "true":
                return HTMLResponse(content="", headers={"HX-Redirect": "/login"})
            return RedirectResponse(url="/login", status_code=303)

        now_utc = datetime.now(timezone.utc)
        current_date = now_utc.date()
        yesterday_date = current_date - timedelta(days=1)
        tomorrow_date = current_date + timedelta(days=1)

        filter_val = filter if isinstance(filter, str) else ""
        date_filter_val = date_filter if isinstance(date_filter, str) else ""
        active_date_filter = (filter_val or date_filter_val).lower()
        if isinstance(date, str) and date.strip():
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

        # 1. Query matches based on active date filter
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
        elif active_date_filter == "live":
            start_dt = datetime.combine(current_date, datetime.min.time())
            end_dt = start_dt + timedelta(days=1)
            match_query = match_query.where(Match.kickoff_time >= start_dt, Match.kickoff_time < end_dt)
        elif active_date_filter == "upcoming":
            match_query = match_query.where(
                Match.kickoff_time >= now_utc.replace(tzinfo=None),
                Match.status == MatchStatus.SCHEDULED,
            )
        elif active_date_filter == "all":
            pass
        else:
            try:
                custom_d = datetime.strptime(active_date_filter, "%Y-%m-%d").date()
                start_dt = datetime.combine(custom_d, datetime.min.time())
                end_dt = start_dt + timedelta(days=1)
                match_query = match_query.where(Match.kickoff_time >= start_dt, Match.kickoff_time < end_dt)
            except Exception:
                start_dt = datetime.combine(current_date, datetime.min.time())
                end_dt = start_dt + timedelta(days=1)
                match_query = match_query.where(Match.kickoff_time >= start_dt, Match.kickoff_time < end_dt)

        clean_sport = None if (not sport or sport.lower() == "all") else sport.lower()

        # Compute sport match counts for the active date across all sports
        sport_match_counts = {"football": 0, "basketball": 0, "tennis": 0}
        date_where = []
        if active_date_filter == "today":
            start_dt = datetime.combine(current_date, datetime.min.time())
            end_dt = start_dt + timedelta(days=1)
            date_where = [Match.kickoff_time >= start_dt, Match.kickoff_time < end_dt]
        elif active_date_filter == "yesterday":
            start_dt = datetime.combine(yesterday_date, datetime.min.time())
            end_dt = start_dt + timedelta(days=1)
            date_where = [Match.kickoff_time >= start_dt, Match.kickoff_time < end_dt]
        elif active_date_filter == "tomorrow":
            start_dt = datetime.combine(tomorrow_date, datetime.min.time())
            end_dt = start_dt + timedelta(days=1)
            date_where = [Match.kickoff_time >= start_dt, Match.kickoff_time < end_dt]
        elif active_date_filter == "live":
            start_dt = datetime.combine(current_date, datetime.min.time())
            end_dt = start_dt + timedelta(days=1)
            date_where = [Match.kickoff_time >= start_dt, Match.kickoff_time < end_dt]
        elif active_date_filter == "upcoming":
            date_where = [
                Match.kickoff_time >= now_utc.replace(tzinfo=None),
                Match.status == MatchStatus.SCHEDULED,
            ]
        elif active_date_filter == "all":
            date_where = []
        else:
            try:
                custom_d = datetime.strptime(active_date_filter, "%Y-%m-%d").date()
                start_dt = datetime.combine(custom_d, datetime.min.time())
                end_dt = start_dt + timedelta(days=1)
                date_where = [Match.kickoff_time >= start_dt, Match.kickoff_time < end_dt]
            except Exception:
                start_dt = datetime.combine(current_date, datetime.min.time())
                end_dt = start_dt + timedelta(days=1)
                date_where = [Match.kickoff_time >= start_dt, Match.kickoff_time < end_dt]

        count_q = (
            select(Sport.name, func.count(Match.id))
            .join(Match, Match.sport_id == Sport.id)
        )
        if date_where:
            count_q = count_q.where(*date_where)
        count_q = count_q.group_by(Sport.name)
        try:
            for sp_name, sp_cnt in db.execute(count_q).all():
                k = (sp_name or "").lower()
                if k in sport_match_counts:
                    sport_match_counts[k] = sp_cnt
                elif "foot" in k or "soccer" in k:
                    sport_match_counts["football"] += sp_cnt
                elif "basket" in k or "nba" in k:
                    sport_match_counts["basketball"] += sp_cnt
                elif "ten" in k:
                    sport_match_counts["tennis"] += sp_cnt
        except Exception as e:
            log.warning("Could not execute sport count query: %s", e)

        if clean_sport:
            match_query = match_query.join(Match.sport).where(
                or_(
                    func.lower(Sport.name) == clean_sport,
                    func.lower(Sport.slug) == clean_sport,
                    func.lower(Sport.name).contains(clean_sport),
                )
            )

        scheduled_matches = db.scalars(match_query).all()

        # Deduplicate records
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

        sport_match_counts = {"football": 0, "basketball": 0, "tennis": 0}
        for m in deduped_matches:
            sp_name = (m.sport.name if m.sport else "").lower()
            if "foot" in sp_name or "soccer" in sp_name:
                sport_match_counts["football"] += 1
            elif "basket" in sp_name or "nba" in sp_name:
                sport_match_counts["basketball"] += 1
            elif "ten" in sp_name:
                sport_match_counts["tennis"] += 1
        sport_match_counts["all"] = len(deduped_matches)

        filtered_raw_matches = []
        for m in deduped_matches:
            sp_name = (m.sport.name if m.sport else "").lower()
            m_sport_key = "basketball" if "basket" in sp_name or "nba" in sp_name else ("tennis" if "ten" in sp_name else "football")
            if clean_sport and m_sport_key != clean_sport:
                continue
            if q:
                q_clean = q.strip().lower()
                h_n = (m.home_entity.name if m.home_entity else "").lower()
                a_n = (m.away_entity.name if m.away_entity else "").lower()
                c_n = (m.competition_name or "").lower()
                if q_clean not in h_n and q_clean not in a_n and q_clean not in c_n:
                    continue
            if active_date_filter == "live":
                final_s = m.final_score or {}
                is_m_live = (m.status == MatchStatus.LIVE or bool(final_s.get("is_live")) or str(final_s.get("status", "")).lower() == "live")
                if not is_m_live:
                    continue
            filtered_raw_matches.append(m)

        filtered_raw_matches.sort(
            key=lambda x: (
                x.status != MatchStatus.LIVE,
                x.kickoff_time or datetime.max,
            )
        )

        matches_to_enrich = filtered_raw_matches[:60]
        matches_map: dict[int, dict[str, Any]] = {}
        sport_safe_counts = {"football": 0, "basketball": 0, "tennis": 0}
        total_safe_predictions = 0

        feature_builder = FeatureBuilder()
        for m in matches_to_enrich:
            m_dict, active_plays = enrich_match_for_ui(m, db, feature_builder)
            matches_map[m.id] = m_dict
            sport_key = m_dict["sport_key"]
            if sport_key in sport_safe_counts:
                sport_safe_counts[sport_key] += len(active_plays)
            total_safe_predictions += len(active_plays)

        all_matches = list(matches_map.values())
        total_matches_count = sum(sport_match_counts.values()) if not clean_sport else sport_match_counts.get(clean_sport, len(all_matches))
        total_safe_matches_count = sum(1 for m in all_matches if m["has_safe_pick"])

        filtered_matches = []
        for m in all_matches:
            if safe_only and not m["has_safe_pick"]:
                continue
            filtered_matches.append(m)

        filtered_matches.sort(
            key=lambda x: (
                not x.get("is_live"),
                not x["has_safe_pick"],
                -x["max_confidence"],
                x["kickoff_time"] or datetime.max,
            )
        )

        matches_json_list = []
        for m in all_matches:
            m_dict = dict(m)
            if isinstance(m_dict.get("kickoff_time"), datetime):
                m_dict["kickoff_time"] = m_dict["kickoff_time"].isoformat()
            matches_json_list.append(m_dict)

        matches_json = json.dumps(matches_json_list)
        retrain_info = get_model_retrain_metadata()
        last_trained_date = retrain_info.get("last_trained_display", "Today")

        current_day_num = str(current_date.day)
        sport_param = f"&sport={sport}" if (sport and sport != "all") else ""
        safe_param = "&safe_only=true" if safe_only else ""

        if active_date_filter in ["today", ""]:
            date_display_label = "Today"
            prev_date_url = f"/matches?filter=yesterday{sport_param}{safe_param}"
            next_date_url = f"/matches?filter=tomorrow{sport_param}{safe_param}"
        elif active_date_filter == "yesterday":
            date_display_label = "Yesterday"
            two_days_ago = (current_date - timedelta(days=2)).strftime("%Y-%m-%d")
            prev_date_url = f"/matches?date={two_days_ago}{sport_param}{safe_param}"
            next_date_url = f"/matches?filter=today{sport_param}{safe_param}"
        elif active_date_filter == "tomorrow":
            date_display_label = "Tomorrow"
            prev_date_url = f"/matches?filter=today{sport_param}{safe_param}"
            in_two_days = (current_date + timedelta(days=2)).strftime("%Y-%m-%d")
            next_date_url = f"/matches?date={in_two_days}{sport_param}{safe_param}"
        elif active_date_filter == "live":
            date_display_label = "Live"
            prev_date_url = f"/matches?filter=yesterday{sport_param}{safe_param}"
            next_date_url = f"/matches?filter=tomorrow{sport_param}{safe_param}"
        else:
            try:
                parsed_d = datetime.strptime(active_date_filter, "%Y-%m-%d").date()
                if parsed_d == current_date:
                    date_display_label = "Today"
                elif parsed_d == yesterday_date:
                    date_display_label = "Yesterday"
                elif parsed_d == tomorrow_date:
                    date_display_label = "Tomorrow"
                else:
                    date_display_label = parsed_d.strftime("%b %d")
                p_day = (parsed_d - timedelta(days=1)).strftime("%Y-%m-%d")
                n_day = (parsed_d + timedelta(days=1)).strftime("%Y-%m-%d")
                prev_date_url = f"/matches?date={p_day}{sport_param}{safe_param}"
                next_date_url = f"/matches?date={n_day}{sport_param}{safe_param}"
            except Exception:
                date_display_label = "Today"
                prev_date_url = f"/matches?filter=yesterday{sport_param}{safe_param}"
                next_date_url = f"/matches?filter=tomorrow{sport_param}{safe_param}"

        is_htmx = request.headers.get("HX-Request") == "true"
        hx_target = request.headers.get("HX-Target")
        base_template = "empty_base.html" if is_htmx else "base.html"

        context = {
            "request": request,
            "current_user": current_user,
            "base_template": base_template,
            "is_htmx": is_htmx,
            "matches": filtered_matches,
            "live_matches": [
                m for m in all_matches
                if m.get("is_live") and (not clean_sport or m.get("sport_key") == clean_sport)
            ],
            "matches_json": matches_json,
            "selected_match": filtered_matches[0] if filtered_matches else None,
            "total_matches": total_matches_count,
            "total_predictions": total_safe_predictions,
            "total_safe_picks": total_safe_predictions,
            "total_safe_matches": total_safe_matches_count,
            "sport_match_counts": sport_match_counts,
            "sport_safe_counts": sport_safe_counts,
            "selected_sport": (sport or "all").lower(),
            "safe_only": safe_only,
            "search_query": q or "",
            "now_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "last_trained_date": last_trained_date,
            "retrain_info": retrain_info,
            "active_date_filter": active_date_filter,
            "current_date_str": current_date.strftime("%Y-%m-%d"),
            "yesterday_date_str": yesterday_date.strftime("%Y-%m-%d"),
            "tomorrow_date_str": tomorrow_date.strftime("%Y-%m-%d"),
            "calendar_days": build_calendar_days(current_date, active_date_filter),
            "date_display_label": date_display_label,
            "prev_date_url": prev_date_url,
            "next_date_url": next_date_url,
            "current_day_num": current_day_num,
            "sync_status": get_sync_status(),
        }

        if is_htmx and hx_target in ["match-board-container", "match-board-content", "match-board"]:
            return templates.TemplateResponse(
                request=request,
                name="partials/match_board.html",
                context=context,
            )

        return templates.TemplateResponse(
            request=request,
            name="matches.html",
            context=context,
        )

    @router.get("/match/{match_id}", response_class=HTMLResponse)
    @router.get("/ui/match/{match_id}", response_class=HTMLResponse)
    def get_match_view(
        request: Request,
        match_id: int,
        db: Session = Depends(get_db),
        current_user: Optional[User] = Depends(get_current_user),
    ):
        """Serve the dedicated full-width Match Analysis Page."""
        if not current_user:
            if request.headers.get("HX-Request") == "true":
                return HTMLResponse(content="", headers={"HX-Redirect": "/login"})
            return RedirectResponse(url="/login", status_code=303)

        match = db.scalar(
            select(Match)
            .options(
                joinedload(Match.sport),
                joinedload(Match.home_entity),
                joinedload(Match.away_entity),
            )
            .where(Match.id == match_id)
        )
        if not match:
            raise HTTPException(status_code=404, detail="Match not found")

        m_dict, active_plays = enrich_match_for_ui(match, db)
        is_htmx = request.headers.get("HX-Request") == "true"
        base_template = "empty_base.html" if is_htmx else "base.html"

        home_entity = match.home_entity
        away_entity = match.away_entity

        home_stats: dict[str, Any] = {}
        away_stats: dict[str, Any] = {}

        if home_entity:
            try:
                home_stats = synthesize_entity_telemetry(home_entity, db, current_match=match)
                db.commit()
            except Exception as e:
                log.warning("Could not synthesize home telemetry for %s: %s", home_entity.name, e)
                db.rollback()
            if not home_stats:
                home_stats = {
                    "form_string": home_entity.form_string or "WWWDW",
                    "goals_scored_per_game": home_entity.goals_scored_per_game or 1.4,
                    "goals_conceded_per_game": home_entity.goals_conceded_per_game or 0.8,
                    "over_2_5_pct": home_entity.over_2_5_pct or 50.0,
                    "over_1_5_pct": min(100.0, round((home_entity.over_2_5_pct or 50.0) + 25.0, 1)),
                    "clean_sheets": home_entity.clean_sheets or 0,
                    "rolling_xga_per_game": home_entity.rolling_xga_per_game or 1.2,
                    "rolling_shots_on_target_conceded": home_entity.rolling_shots_on_target_conceded or 4.0,
                    "big_chances_ratio": home_entity.big_chances_ratio or 1.0,
                    "first_half_clean_sheet_pct": home_entity.first_half_clean_sheet_pct or 50.0,
                    "stalemate_index": home_entity.stalemate_index or 0.0,
                    "foul_card_intensity": home_entity.foul_card_intensity or 2.0,
                    "promotion_tax_applied": bool(getattr(home_entity, "promotion_tax_applied", False)),
                }
        else:
            home_stats = {
                "form_string": "WWWDW",
                "goals_scored_per_game": 1.4,
                "goals_conceded_per_game": 0.8,
                "over_2_5_pct": 50.0,
                "over_1_5_pct": 75.0,
                "clean_sheets": 2,
                "rolling_xga_per_game": 1.2,
                "rolling_shots_on_target_conceded": 4.0,
                "big_chances_ratio": 1.0,
                "first_half_clean_sheet_pct": 50.0,
                "stalemate_index": 0.0,
                "foul_card_intensity": 2.0,
                "promotion_tax_applied": False,
            }

        if away_entity:
            try:
                away_stats = synthesize_entity_telemetry(away_entity, db, current_match=match)
                db.commit()
            except Exception as e:
                log.warning("Could not synthesize away telemetry for %s: %s", away_entity.name, e)
                db.rollback()
            if not away_stats:
                away_stats = {
                    "form_string": away_entity.form_string or "LDWWL",
                    "goals_scored_per_game": away_entity.goals_scored_per_game or 1.2,
                    "goals_conceded_per_game": away_entity.goals_conceded_per_game or 1.0,
                    "over_2_5_pct": away_entity.over_2_5_pct or 40.0,
                    "over_1_5_pct": min(100.0, round((away_entity.over_2_5_pct or 40.0) + 25.0, 1)),
                    "clean_sheets": away_entity.clean_sheets or 0,
                    "rolling_xga_per_game": away_entity.rolling_xga_per_game or 1.2,
                    "rolling_shots_on_target_conceded": away_entity.rolling_shots_on_target_conceded or 4.0,
                    "big_chances_ratio": away_entity.big_chances_ratio or 1.0,
                    "first_half_clean_sheet_pct": away_entity.first_half_clean_sheet_pct or 50.0,
                    "stalemate_index": away_entity.stalemate_index or 0.0,
                    "foul_card_intensity": away_entity.foul_card_intensity or 2.0,
                    "promotion_tax_applied": bool(getattr(away_entity, "promotion_tax_applied", False)),
                }
        else:
            away_stats = {
                "form_string": "LDWWL",
                "goals_scored_per_game": 1.2,
                "goals_conceded_per_game": 1.0,
                "over_2_5_pct": 40.0,
                "over_1_5_pct": 70.0,
                "clean_sheets": 1,
                "rolling_xga_per_game": 1.2,
                "rolling_shots_on_target_conceded": 4.0,
                "big_chances_ratio": 1.0,
                "first_half_clean_sheet_pct": 50.0,
                "stalemate_index": 0.0,
                "foul_card_intensity": 2.0,
            }

        if "over_1_5_pct" not in home_stats or not home_stats["over_1_5_pct"]:
            home_stats["over_1_5_pct"] = min(100.0, round((home_stats.get("over_2_5_pct") or 50.0) + 25.0, 1))
        if "over_1_5_pct" not in away_stats or not away_stats["over_1_5_pct"]:
            away_stats["over_1_5_pct"] = min(100.0, round((away_stats.get("over_2_5_pct") or 40.0) + 25.0, 1))

        h_gf = float(home_stats.get("goals_scored_per_game") or 1.4)
        a_gf = float(away_stats.get("goals_scored_per_game") or 1.2)
        tot_att = max(0.1, h_gf + a_gf)
        home_attack_weight = int(round((h_gf / tot_att) * 100))
        away_attack_weight = 100 - home_attack_weight

        h_ga = float(home_stats.get("goals_conceded_per_game") or 0.8)
        a_ga = float(away_stats.get("goals_conceded_per_game") or 1.0)
        h_def_idx = max(0.2, 3.2 - h_ga)
        a_def_idx = max(0.2, 3.2 - a_ga)
        tot_def = max(0.1, h_def_idx + a_def_idx)
        home_defense_weight = int(round((h_def_idx / tot_def) * 100))
        away_defense_weight = 100 - home_defense_weight

        home_attack_rating = int(round(min(99, max(45, (h_gf / 2.5) * 80 + 18))))
        away_attack_rating = int(round(min(99, max(45, (a_gf / 2.5) * 80 + 18))))
        home_defense_rating = int(round(min(99, max(45, (h_def_idx / 3.0) * 80 + 18))))
        away_defense_rating = int(round(min(99, max(45, (a_def_idx / 3.0) * 80 + 18))))

        def _calc_fp(fs: str) -> int:
            return sum(3 if c == "W" else 1 if c == "D" else 0 for c in (fs or ""))

        h_form_pts = _calc_fp(home_stats.get("form_string", "WWWDW"))
        a_form_pts = _calc_fp(away_stats.get("form_string", "LDWWL"))
        tot_form = max(1, h_form_pts + a_form_pts)
        home_form_weight = int(round((h_form_pts / tot_form) * 100))
        away_form_weight = 100 - home_form_weight

        home_overall_weight = int(round((home_attack_weight * 0.45) + (home_defense_weight * 0.35) + (home_form_weight * 0.20)))
        away_overall_weight = 100 - home_overall_weight

        bivariate = m_dict.get("bivariate") or compute_bivariate_poisson_engine(match, session=db)
        home_win_pct = bivariate.get("home_win_pct", 52)
        draw_pct = bivariate.get("draw_pct", 26)
        away_win_pct = bivariate.get("away_win_pct", 22)
        promotion_tax_applied = bool(m_dict.get("promotion_tax_applied"))

        sport_match_counts = {"football": 0, "basketball": 0, "tennis": 0}
        try:
            count_q = (
                select(Sport.name, func.count(Match.id))
                .join(Match, Match.sport_id == Sport.id)
                .group_by(Sport.name)
            )
            for sp_name, sp_cnt in db.execute(count_q).all():
                k = (sp_name or "").lower()
                if k in sport_match_counts:
                    sport_match_counts[k] = sp_cnt
                elif "foot" in k or "soccer" in k:
                    sport_match_counts["football"] += sp_cnt
                elif "basket" in k or "nba" in k:
                    sport_match_counts["basketball"] += sp_cnt
                elif "ten" in k:
                    sport_match_counts["tennis"] += sp_cnt
        except Exception as e:
            log.warning("Could not execute sport count query: %s", e)

        total_matches_count = sum(sport_match_counts.values())

        return templates.TemplateResponse(
            request=request,
            name="match_detail.html",
            context={
                "request": request,
                "match": m_dict,
                "predictions": active_plays,
                "current_user": current_user,
                "base_template": base_template,
                "is_htmx": is_htmx,
                "selected_sport": m_dict.get("sport_key", "all"),
                "sport_match_counts": sport_match_counts,
                "total_matches": total_matches_count,
                "home_entity": home_entity,
                "away_entity": away_entity,
                "home_stats": home_stats,
                "away_stats": away_stats,
                "home_attack_weight": home_attack_weight,
                "away_attack_weight": away_attack_weight,
                "home_defense_weight": home_defense_weight,
                "away_defense_weight": away_defense_weight,
                "home_overall_weight": home_overall_weight,
                "away_overall_weight": away_overall_weight,
                "home_form_weight": home_form_weight,
                "away_form_weight": away_form_weight,
                "home_attack_rating": home_attack_rating,
                "away_attack_rating": away_attack_rating,
                "home_defense_rating": home_defense_rating,
                "away_defense_rating": away_defense_rating,
                "bivariate": bivariate,
                "home_win_pct": home_win_pct,
                "draw_pct": draw_pct,
                "away_win_pct": away_win_pct,
                "promotion_tax_applied": promotion_tax_applied,
            },
        )

    @router.get("/ui/feed", response_class=HTMLResponse)
    def get_feed_partial(
        request: Request,
        sport: Optional[str] = Query(None),
        safe_only: bool = Query(False),
        q: Optional[str] = Query(None),
        filter: Optional[str] = Query(None),
        db: Session = Depends(get_db),
        current_user: Optional[User] = Depends(get_current_user),
    ):
        """Legacy feed redirecting to /matches."""
        target_url = "/matches"
        params = []
        if sport and sport != "all":
            params.append(f"sport={sport}")
        if filter:
            params.append(f"filter={filter}")
        if safe_only:
            params.append("safe_only=true")
        if q:
            params.append(f"q={q}")
        if params:
            target_url += "?" + "&".join(params)

        if request.headers.get("HX-Request") == "true":
            return HTMLResponse(content="", headers={"HX-Location": f'{{"path": "{target_url}", "target": "#app-content"}}'})
        return RedirectResponse(url=target_url, status_code=303)

    @router.get("/ui/workspace", response_class=HTMLResponse)
    def get_workspace_partial(
        request: Request,
        sport: Optional[str] = Query(None),
        safe_only: bool = Query(False),
        q: Optional[str] = Query(None),
        filter: Optional[str] = Query(None),
        date_filter: Optional[str] = Query(None),
        date: Optional[str] = Query(None),
        db: Session = Depends(get_db),
        current_user: Optional[User] = Depends(get_current_user),
    ):
        """Legacy workspace redirecting to /matches."""
        if not current_user:
            if request.headers.get("HX-Request") == "true":
                return HTMLResponse(content="", headers={"HX-Redirect": "/login"})
            return RedirectResponse(url="/login", status_code=303)

        target_url = "/matches"
        params = []
        if sport and sport != "all":
            params.append(f"sport={sport}")
        if filter:
            params.append(f"filter={filter}")
        if date_filter:
            params.append(f"date_filter={date_filter}")
        if safe_only:
            params.append("safe_only=true")
        if q:
            params.append(f"q={q}")
        if params:
            target_url += "?" + "&".join(params)

        if request.headers.get("HX-Request") == "true":
            return HTMLResponse(content="", headers={"HX-Location": f'{{"path": "{target_url}", "target": "#app-content"}}'})
        return RedirectResponse(url=target_url, status_code=303)

    @router.get("/search", response_class=HTMLResponse)
    def live_search(
        request: Request,
        q: Optional[str] = Query(None, description="Search term for entity or fixture"),
        db: Session = Depends(get_db),
        current_user: Optional[User] = Depends(get_current_user),
    ):
        """Live fixture and team search endpoint."""
        clean_q = (q or "").strip()
        entities = []
        matches = []

        if clean_q:
            entities_stmt = (
                select(Entity)
                .options(joinedload(Entity.sport))
                .where(Entity.name.ilike(f"%{clean_q}%"))
                .order_by(Entity.name.asc())
                .limit(5)
            )
            entities = db.scalars(entities_stmt).unique().all()

            h_alias = aliased(Entity)
            a_alias = aliased(Entity)
            matches_stmt = (
                select(Match)
                .options(
                    joinedload(Match.home_entity),
                    joinedload(Match.away_entity),
                    joinedload(Match.sport),
                )
                .outerjoin(h_alias, Match.home_entity_id == h_alias.id)
                .outerjoin(a_alias, Match.away_entity_id == a_alias.id)
                .where(
                    or_(
                        h_alias.name.ilike(f"%{clean_q}%"),
                        a_alias.name.ilike(f"%{clean_q}%"),
                        Match.competition_name.ilike(f"%{clean_q}%"),
                    )
                )
                .order_by(Match.kickoff_time.desc())
                .limit(6)
            )
            matches = db.scalars(matches_stmt).unique().all()

        context = {
            "request": request,
            "query": clean_q,
            "entities": entities,
            "matches": matches,
            "total_count": len(entities) + len(matches),
            "current_user": current_user,
        }

        return templates.TemplateResponse(
            request=request,
            name="search_results.html",
            context=context,
        )

    @router.get("/api/matches/live", response_class=HTMLResponse)
    def get_live_matches_partial(
        request: Request,
        sport: Optional[str] = Query(None, description="Filter live matches by sport"),
        db: Session = Depends(get_db),
    ):
        """Rendered HTML partial of live match rows for HTMX auto-polling."""
        query = (
            select(Match)
            .options(
                joinedload(Match.sport),
                joinedload(Match.home_entity),
                joinedload(Match.away_entity),
            )
            .where(Match.status == MatchStatus.LIVE)
            .order_by(Match.kickoff_time.desc())
        )

        matches_found = db.scalars(query).all()
        clean_sport = (sport or "").strip().lower()

        enriched_live = []
        feature_builder = FeatureBuilder()
        for m in matches_found:
            m_dict, _ = enrich_match_for_ui(m, db, feature_builder)
            if not m_dict.get("is_live"):
                continue
            if clean_sport and clean_sport != "all" and m_dict.get("sport_key") != clean_sport:
                continue
            enriched_live.append(m_dict)

        filter_param = (request.query_params.get("filter") or "").strip().lower()

        return templates.TemplateResponse(
            request=request,
            name="partials/live_matches.html",
            context={
                "request": request,
                "live_matches": enriched_live,
                "matches": enriched_live,
                "active_date_filter": filter_param,
            },
        )

    return router
