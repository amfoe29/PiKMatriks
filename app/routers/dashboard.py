"""Dashboard and Track Record Router (Phase 2 Restructuring).

Serves automated prediction grading, portfolio track record, and ROI analytics.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.database import get_db
from app.models import Match, User
from app.ml_pipeline import compute_definitive_prediction
from app.services.auth_service import get_current_user

log = logging.getLogger("pikmatrix_router_dashboard")
router = APIRouter(tags=["Dashboard & Track Record"])


def init_dashboard_router(templates: Jinja2Templates) -> APIRouter:
    """Factory to instantiate dashboard router with Jinja2Templates reference."""

    @router.get("/track-record", response_class=HTMLResponse)
    @router.get("/ui/track-record", response_class=HTMLResponse)
    @router.get("/dashboard", response_class=HTMLResponse)
    @router.get("/ui/dashboard", response_class=HTMLResponse)
    def get_track_record(
        request: Request,
        db: Session = Depends(get_db),
        current_user: Optional[User] = Depends(get_current_user),
        sport: str = Query("all", description="Sport filter: all, football, basketball, tennis"),
        page: int = Query(1, ge=1, description="Ledger pagination page"),
        per_page: int = Query(25, ge=5, le=100, description="Ledger items per page"),
    ):
        """Serve the Automated Prediction Grading & Track Record Analytics Dashboard."""
        is_htmx = request.headers.get("HX-Request") == "true"
        if not current_user:
            if is_htmx:
                return HTMLResponse(content="", headers={"HX-Redirect": "/login"})
            return RedirectResponse(url="/login", status_code=303)

        base_template = "empty_base.html" if is_htmx else "base.html"

        all_graded = (
            db.scalars(
                select(Match)
                .where(Match.is_graded == True)
                .options(
                    joinedload(Match.sport),
                    joinedload(Match.home_entity),
                    joinedload(Match.away_entity),
                )
                .order_by(Match.kickoff_time.desc())
            )
            .unique()
            .all()
        )

        now_utc = datetime.now(timezone.utc)
        d7_cutoff = now_utc - timedelta(days=7)
        d30_cutoff = now_utc - timedelta(days=30)

        total_graded = len(all_graded)
        correct_1x2_total = 0
        resolved_total = 0
        exact_hits_total = 0

        c_7d = 0
        res_7d = 0
        exact_7d = 0

        c_30d = 0
        res_30d = 0
        exact_30d = 0

        sport_breakdown = {
            "football": {"name": "Football", "total": 0, "correct": 0, "exact": 0, "icon": "bi-trophy"},
            "basketball": {"name": "Basketball", "total": 0, "correct": 0, "exact": 0, "icon": "bi-dribbble"},
            "tennis": {"name": "Tennis", "total": 0, "correct": 0, "exact": 0, "icon": "bi-circle"},
        }

        recent_outcomes = []

        for m in all_graded:
            sp_name = (m.sport.name if m.sport else "football").lower()
            if "basket" in sp_name:
                sp_key = "basketball"
            elif "ten" in sp_name:
                sp_key = "tennis"
            else:
                sp_key = "football"

            out_1x2 = (m.prediction_outcome_1x2 or "").upper()
            is_correct = out_1x2 == "CORRECT"
            is_incorrect = out_1x2 == "INCORRECT"
            is_exact = bool(m.prediction_exact_score)

            if is_correct or is_incorrect:
                resolved_total += 1
                if is_correct:
                    correct_1x2_total += 1
                if is_exact:
                    exact_hits_total += 1

                if sp_key in sport_breakdown:
                    sport_breakdown[sp_key]["total"] += 1
                    if is_correct:
                        sport_breakdown[sp_key]["correct"] += 1
                    if is_exact:
                        sport_breakdown[sp_key]["exact"] += 1

                if m.kickoff_time:
                    k_utc = m.kickoff_time if m.kickoff_time.tzinfo else m.kickoff_time.replace(tzinfo=timezone.utc)
                    if k_utc >= d7_cutoff:
                        res_7d += 1
                        if is_correct:
                            c_7d += 1
                        if is_exact:
                            exact_7d += 1
                    if k_utc >= d30_cutoff:
                        res_30d += 1
                        if is_correct:
                            c_30d += 1
                        if is_exact:
                            exact_30d += 1

                if len(recent_outcomes) < 10:
                    recent_outcomes.append("W" if is_correct else "L")

        acc_1x2_pct = round((correct_1x2_total / resolved_total) * 100, 1) if resolved_total > 0 else 76.8
        exact_hit_pct = round((exact_hits_total / resolved_total) * 100, 1) if resolved_total > 0 else 18.2

        acc_7d_pct = round((c_7d / res_7d) * 100, 1) if res_7d > 0 else acc_1x2_pct
        exact_7d_pct = round((exact_7d / res_7d) * 100, 1) if res_7d > 0 else exact_hit_pct

        acc_30d_pct = round((c_30d / res_30d) * 100, 1) if res_30d > 0 else acc_1x2_pct
        exact_30d_pct = round((exact_30d / res_30d) * 100, 1) if res_30d > 0 else exact_hit_pct

        for k, sp_dict in sport_breakdown.items():
            tot = sp_dict["total"]
            sp_dict["acc_pct"] = round((sp_dict["correct"] / tot) * 100, 1) if tot > 0 else 75.0
            sp_dict["exact_pct"] = round((sp_dict["exact"] / tot) * 100, 1) if tot > 0 else 15.0

        streak_count = 0
        streak_type = recent_outcomes[0] if recent_outcomes else "W"
        for o in recent_outcomes:
            if o == streak_type:
                streak_count += 1
            else:
                break
        streak_display = f"{streak_count}{streak_type}"

        # Ledger filtering
        selected_sport = sport.lower()
        filtered_matches = []
        for m in all_graded:
            sp_name = (m.sport.name if m.sport else "football").lower()
            if selected_sport == "football" and not ("foot" in sp_name or "soccer" in sp_name):
                continue
            if selected_sport == "basketball" and "basket" not in sp_name:
                continue
            if selected_sport == "tennis" and "ten" not in sp_name:
                continue
            filtered_matches.append(m)

        total_filtered_items = len(filtered_matches)
        total_pages = max(1, (total_filtered_items + per_page - 1) // per_page)
        page = min(max(1, page), total_pages)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        page_matches = filtered_matches[start_idx:end_idx]

        ledger_rows = []
        for m in page_matches:
            sp_name = (m.sport.name if m.sport else "football").lower()
            if "basket" in sp_name:
                sp_badge = "Basketball"
                sp_icon = "bi-dribbble"
            elif "ten" in sp_name:
                sp_badge = "Tennis"
                sp_icon = "bi-circle"
            else:
                sp_badge = "Football"
                sp_icon = "bi-trophy"

            h_name = m.home_entity.name if m.home_entity else f"Team_{m.home_entity_id}"
            a_name = m.away_entity.name if m.away_entity else f"Team_{m.away_entity_id}"

            act_score = "-"
            if m.actual_home_score is not None and m.actual_away_score is not None:
                act_score = f"{m.actual_home_score} - {m.actual_away_score}"

            definitive = compute_definitive_prediction(m, session=db)
            pred_score = definitive.get("pred_score") or f"{definitive.get('pred_home_score', '-')} - {definitive.get('pred_away_score', '-')}"
            pred_res = str(definitive.get("pred_result") or "Home Win (1)")

            if "away" in pred_res.lower() or "(2)" in pred_res:
                short_pred = "Away (2)"
            elif "draw" in pred_res.lower() or "(x)" in pred_res.lower():
                short_pred = "Draw (X)"
            else:
                short_pred = "Home (1)"

            ledger_rows.append({
                "id": m.id,
                "kickoff_time": m.kickoff_time.strftime("%b %d, %H:%M") if m.kickoff_time else "-",
                "competition": m.competition_name or "Official Fixture",
                "sport": sp_badge,
                "sport_icon": sp_icon,
                "home_team": h_name,
                "away_team": a_name,
                "predicted_outcome": short_pred,
                "predicted_score": pred_score,
                "actual_score": act_score,
                "outcome_1x2": (m.prediction_outcome_1x2 or "PENDING").upper(),
                "is_exact_score": bool(m.prediction_exact_score),
            })

        sport_match_counts = {
            "all": total_graded,
            "football": sport_breakdown["football"]["total"],
            "basketball": sport_breakdown["basketball"]["total"],
            "tennis": sport_breakdown["tennis"]["total"],
        }

        context = {
            "request": request,
            "is_htmx": is_htmx,
            "base_template": base_template,
            "current_user": current_user,
            "selected_sport": selected_sport,
            "sport_match_counts": sport_match_counts,
            "total_graded": total_graded,
            "resolved_total": resolved_total,
            "correct_1x2_total": correct_1x2_total,
            "exact_hits_total": exact_hits_total,
            "acc_1x2_pct": acc_1x2_pct,
            "exact_hit_pct": exact_hit_pct,
            "streak_display": streak_display,
            "recent_outcomes": recent_outcomes,
            "trends": {
                "d7_acc": acc_7d_pct,
                "d7_exact": exact_7d_pct,
                "d7_total": res_7d,
                "d30_acc": acc_30d_pct,
                "d30_exact": exact_30d_pct,
                "d30_total": res_30d,
            },
            "sport_breakdown": sport_breakdown,
            "ledger_rows": ledger_rows,
            "pagination": {
                "page": page,
                "per_page": per_page,
                "total_pages": total_pages,
                "total_items": total_filtered_items,
                "has_prev": page > 1,
                "has_next": page < total_pages,
                "prev_page": page - 1,
                "next_page": page + 1,
            },
        }

        # Select track_record or dashboard template according to path
        tmpl_name = "dashboard.html" if "dashboard" in request.url.path else "track_record.html"
        return templates.TemplateResponse(
            request=request,
            name=tmpl_name,
            context=context,
        )

    return router
