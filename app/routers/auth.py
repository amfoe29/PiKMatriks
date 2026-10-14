"""Authentication Router (Phase 2 Restructuring).

Handles user login, registration, logout, and credential validation.
"""

from __future__ import annotations

import logging
from typing import Optional
from urllib.parse import parse_qs

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.auth import create_access_token, get_password_hash, verify_password
from app.database import get_db
from app.models import User
from app.services.auth_service import get_current_user

log = logging.getLogger("pikmatrix_router_auth")
router = APIRouter(tags=["Authentication"])


async def _extract_auth_credentials(request: Request) -> dict[str, str]:
    """Helper to extract auth fields regardless of Content-Type (JSON, Form, URL-encoded)."""
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            data = await request.json()
            return {
                "email": str(data.get("email") or "").strip().lower(),
                "username": str(data.get("username") or "").strip(),
                "password": str(data.get("password") or ""),
            }
        except Exception:
            return {}

    try:
        form = await request.form()
        return {
            "email": str(form.get("email", "")).strip().lower(),
            "username": str(form.get("username", "")).strip(),
            "password": str(form.get("password", "")),
        }
    except Exception:
        pass

    try:
        raw_body = await request.body()
        parsed = parse_qs(raw_body.decode("utf-8"))
        return {
            "email": parsed.get("email", [""])[0].strip().lower(),
            "username": parsed.get("username", [""])[0].strip(),
            "password": parsed.get("password", [""])[0],
        }
    except Exception:
        return {}


def init_auth_router(templates: Jinja2Templates) -> APIRouter:
    """Factory to instantiate auth router with Jinja2Templates reference."""

    @router.get("/login", response_class=HTMLResponse)
    def login_page(
        request: Request,
        current_user: Optional[User] = Depends(get_current_user),
    ):
        """Render the dedicated PikMatrix login & registration screen."""
        if current_user:
            return RedirectResponse(url="/", status_code=303)
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={"request": request},
        )

    @router.post("/api/register")
    async def register(
        request: Request,
        db: Session = Depends(get_db),
    ):
        """Hash password and create a new User in the database."""
        creds = await _extract_auth_credentials(request)
        email = creds.get("email", "")
        username = creds.get("username", "")
        password = creds.get("password", "")

        is_htmx = request.headers.get("HX-Request") == "true"

        if not email or "@" not in email or "." not in email:
            msg = "Please enter a valid email address."
            if is_htmx:
                return HTMLResponse(
                    f'<div class="alert-auth alert-danger"><span class="badge-status badge-status-error">Error</span> {msg}</div>',
                    status_code=200,
                )
            raise HTTPException(status_code=400, detail=msg)

        if not username:
            username = email.split("@")[0]

        if not password or len(password) < 6:
            msg = "Password must be at least 6 characters long."
            if is_htmx:
                return HTMLResponse(
                    f'<div class="alert-auth alert-danger"><span class="badge-status badge-status-error">Error</span> {msg}</div>',
                    status_code=200,
                )
            raise HTTPException(status_code=400, detail=msg)

        existing_user = db.scalar(select(User).where(func.lower(User.email) == email))
        if existing_user:
            msg = "An account with this email address already exists."
            if is_htmx:
                return HTMLResponse(
                    f'<div class="alert-auth alert-danger"><span class="badge-status badge-status-error">Error</span> {msg}</div>',
                    status_code=200,
                )
            raise HTTPException(status_code=400, detail=msg)

        hashed_pw = get_password_hash(password)
        user = User(
            email=email,
            username=username,
            hashed_password=hashed_pw,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        token = create_access_token(
            data={"sub": str(user.id), "email": user.email, "username": user.username}
        )

        if is_htmx:
            response = HTMLResponse(
                content='<div class="alert-auth alert-success"><span class="badge-status badge-status-success">Success</span> Account created! Launching terminal...</div>',
                headers={"HX-Redirect": "/"},
            )
        else:
            response = JSONResponse(
                status_code=201,
                content={
                    "status": "success",
                    "message": "User created successfully",
                    "access_token": token,
                    "token_type": "bearer",
                    "user": {
                        "id": user.id,
                        "email": user.email,
                        "username": user.username,
                    },
                },
            )

        response.set_cookie(
            key="access_token",
            value=token,
            httponly=True,
            samesite="lax",
            max_age=60 * 60 * 24,
            path="/",
        )
        return response

    @router.post("/api/login")
    async def login(
        request: Request,
        db: Session = Depends(get_db),
    ):
        """Verify credentials, generate JWT, and set HttpOnly access_token cookie."""
        creds = await _extract_auth_credentials(request)
        email = creds.get("email", "")
        password = creds.get("password", "")

        is_htmx = request.headers.get("HX-Request") == "true"

        if not email or not password:
            msg = "Email and password are required."
            if is_htmx:
                return HTMLResponse(
                    f'<div class="alert-auth alert-danger"><span class="badge-status badge-status-error">Error</span> {msg}</div>',
                    status_code=200,
                )
            raise HTTPException(status_code=400, detail=msg)

        user = db.scalar(select(User).where(func.lower(User.email) == email))
        if not user or not verify_password(password, user.hashed_password):
            msg = "Invalid email or password. Please verify your credentials."
            if is_htmx:
                return HTMLResponse(
                    f'<div class="alert-auth alert-danger"><span class="badge-status badge-status-error">Error</span> {msg}</div>',
                    status_code=200,
                )
            raise HTTPException(status_code=401, detail=msg)

        token = create_access_token(
            data={"sub": str(user.id), "email": user.email, "username": user.username}
        )

        if is_htmx:
            response = HTMLResponse(
                content='<div class="alert-auth alert-success"><span class="badge-status badge-status-success">Success</span> Verified! Redirecting to terminal...</div>',
                headers={"HX-Redirect": "/"},
            )
        else:
            response = JSONResponse(
                content={
                    "status": "success",
                    "message": "Login successful",
                    "access_token": token,
                    "token_type": "bearer",
                    "user": {
                        "id": user.id,
                        "email": user.email,
                        "username": user.username,
                    },
                }
            )

        response.set_cookie(
            key="access_token",
            value=token,
            httponly=True,
            samesite="lax",
            max_age=60 * 60 * 24,
            path="/",
        )
        return response

    @router.post("/api/logout")
    @router.get("/logout")
    def logout(request: Request):
        """Clear the access_token cookie and redirect to login screen."""
        is_htmx = request.headers.get("HX-Request") == "true"
        if is_htmx:
            response = HTMLResponse(
                content="",
                headers={"HX-Redirect": "/login"},
            )
        else:
            response = RedirectResponse(url="/login", status_code=303)
            response.headers["HX-Redirect"] = "/login"

        response.delete_cookie(key="access_token", path="/")
        return response

    return router
