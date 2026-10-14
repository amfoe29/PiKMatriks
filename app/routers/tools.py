"""Tools Router (Phase 2 Restructuring).

Serves quantitative betting calculators (+EV, Kelly sizing) and deep prediction breakdowns.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from app.database import get_db
from app.models import Match, Sport, User
from app.ml_pipeline import compute_bivariate_poisson_engine
from app.services.auth_service import get_current_user
from app.services.match_service import get_team_initials

log = logging.getLogger("pikmatrix_router_tools")
router = APIRouter(tags=["Quantitative Tools"])


def init_tools_router(templates: Jinja2Templates) -> APIRouter:
    """Factory to instantiate tools router with Jinja2Templates reference."""

    @router.get("/tools", response_class=HTMLResponse)
    def get_betting_tools(
        request: Request,
        db: Session = Depends(get_db),
        current_user: Optional[User] = Depends(get_current_user),
    ):
        """Serve the quantitative betting tools suite (EV calculator, Kelly Criterion)."""
        if not current_user:
            if request.headers.get("HX-Request") == "true":
                return HTMLResponse(content="", headers={"HX-Redirect": "/login"})
            return RedirectResponse(url="/login", status_code=303)

        is_htmx = request.headers.get("HX-Request") == "true"
        base_template = "empty_base.html" if is_htmx else "base.html"

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

        return templates.TemplateResponse(
            request=request,
            name="tools.html",
            context={
                "request": request,
                "current_user": current_user,
                "base_template": base_template,
                "is_htmx": is_htmx,
                "selected_sport": "all",
                "sport_match_counts": sport_match_counts,
                "total_matches": sum(sport_match_counts.values()),
            },
        )

    @router.get("/api/tools/prediction-breakdown", response_class=HTMLResponse)
    def get_prediction_breakdown(
        request: Request,
        match_id: int = Query(..., description="Target Match ID"),
        db: Session = Depends(get_db),
    ):
        """Deep analytical prediction breakdown drawer using the Bivariate Poisson engine."""
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
            return HTMLResponse(
                "<div class='p-4 text-center text-secondary'>Match not found.</div>",
                status_code=404,
            )

        breakdown = compute_bivariate_poisson_engine(match, session=db)
        home_name = match.home_entity.name if match.home_entity else f"Home_{match.home_entity_id}"
        away_name = match.away_entity.name if match.away_entity else f"Away_{match.away_entity_id}"
        sport_name = match.sport.name.capitalize() if match.sport else "Football"

        return templates.TemplateResponse(
            request=request,
            name="partials/prediction_drawer.html",
            context={
                "request": request,
                "match": match,
                "breakdown": breakdown,
                "home_name": home_name,
                "away_name": away_name,
                "sport_name": sport_name,
                "home_init": get_team_initials(home_name),
                "away_init": get_team_initials(away_name),
            },
        )

    return router
