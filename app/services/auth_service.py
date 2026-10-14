"""Authentication and Session Service (Phase 2 Restructuring).

Handles user resolution from HttpOnly cookies and Bearer tokens,
login/registration validation, and session lifecycle.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.auth import decode_access_token
from app.database import get_db
from app.models import User

log = logging.getLogger("pikmatrix_auth_service")


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> Optional[User]:
    """Dependency that reads the HttpOnly access_token cookie (or Authorization header)
    and returns the active User model."""
    token = request.cookies.get("access_token")
    if not token:
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ", 1)[1]

    if not token:
        return None

    if token.startswith("Bearer "):
        token = token.split(" ", 1)[1]

    payload = decode_access_token(token)
    if not payload:
        return None

    user_id = payload.get("sub")
    if not user_id:
        return None

    try:
        user = db.get(User, int(user_id))
        return user
    except Exception as e:
        log.warning("Failed to resolve user from token id %s: %s", user_id, e)
        return None
