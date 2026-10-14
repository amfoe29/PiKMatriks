"""Profile Router (Phase 2 Restructuring).

Serves user profile overview, preferences, and subscription management.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Match, Sport, User
from app.services.auth_service import get_current_user

log = logging.getLogger("pikmatrix_router_profile")
router = APIRouter(tags=["User Profile"])


def init_profile_router(templates: Jinja2Templates) -> APIRouter:
    """Factory to instantiate profile router with Jinja2Templates reference."""

    @router.get("/profile", response_class=HTMLResponse)
    def get_user_profile(
        request: Request,
        db: Session = Depends(get_db),
        current_user: Optional[User] = Depends(get_current_user),
    ):
        """Serve the dedicated User Profile & Preferences Page."""
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

        display_name = current_user.username if current_user.username else "Amfo Ebenezer"
        user_initials = (display_name[:2]).upper() if display_name else "AE"
        display_email = current_user.email if current_user.email else "amfo.ebenezer@pikmatrix.com"

        return templates.TemplateResponse(
            request=request,
            name="profile.html",
            context={
                "request": request,
                "current_user": current_user,
                "base_template": base_template,
                "is_htmx": is_htmx,
                "display_name": "Amfo Ebenezer",
                "user_initials": "AE",
                "display_email": display_email,
                "selected_sport": "all",
                "sport_match_counts": sport_match_counts,
                "total_matches": sum(sport_match_counts.values()),
            },
        )

    return router
