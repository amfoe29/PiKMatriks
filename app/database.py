"""Database engine, session factory and declarative Base (SQLAlchemy 2.0 style).

Connection resolution order:
    1. ``DATABASE_URL`` environment variable (any SQLAlchemy URL).
    2. ``DB_USER`` / ``DB_PASSWORD`` / ``DB_HOST`` / ``DB_PORT`` / ``DB_NAME``
       -> PostgreSQL via psycopg2.
    3. Fallback: local SQLite file ``sqlite:///./pikmatrix.db`` for offline testing.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator
from typing import Union

from dotenv import load_dotenv
from sqlalchemy import URL, create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()

logger = logging.getLogger(__name__)

SQLITE_FALLBACK_URL = "sqlite:///./pikmatrix.db"


def _env(name: str) -> str | None:
    """Return a stripped env var, treating empty strings as unset."""
    value = os.getenv(name, "").strip()
    return value or None


# Resolve database URL: prioritize DATABASE_URL, then DB_* parts, falling back to SQLite
SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./pikmatrix.db")

# Fallback to DB_* parts if DATABASE_URL was not explicitly set in environment
if SQLALCHEMY_DATABASE_URL == "sqlite:///./pikmatrix.db":
    user, host, name = _env("DB_USER"), _env("DB_HOST"), _env("DB_NAME")
    if user and host and name:
        SQLALCHEMY_DATABASE_URL = str(
            URL.create(
                drivername="postgresql+psycopg2",
                username=user,
                password=_env("DB_PASSWORD"),
                host=host,
                port=int(_env("DB_PORT") or 5432),
                database=name,
            )
        )

# Maintain DATABASE_URL alias for backward compatibility
DATABASE_URL = SQLALCHEMY_DATABASE_URL
_IS_SQLITE = str(SQLALCHEMY_DATABASE_URL).startswith("sqlite")

# Dynamic connection arguments:
# SQLite requires check_same_thread=False for multi-threaded/worker use, while PostgreSQL does not accept it.
connect_args = {"check_same_thread": False, "timeout": 30} if _IS_SQLITE else {}

engine_kwargs: dict[str, Any] = {
    "echo": (_env("DB_ECHO") or "false").lower() == "true",
    "pool_pre_ping": True,
    "connect_args": connect_args,
}

if not _IS_SQLITE:
    # Production-ready PostgreSQL connection pool configuration
    engine_kwargs.update(
        {
            "pool_size": int(os.getenv("DB_POOL_SIZE", "10")),
            "max_overflow": int(os.getenv("DB_MAX_OVERFLOW", "20")),
            "pool_recycle": int(os.getenv("DB_POOL_RECYCLE", "1800")),
        }
    )

engine: Engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    **engine_kwargs,
)


if _IS_SQLITE:

    @event.listens_for(engine, "connect")
    def _enable_sqlite_foreign_keys(dbapi_connection, _connection_record) -> None:
        """SQLite ignores FK constraints unless explicitly enabled per connection."""
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=15000")
        cursor.close()


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def get_db() -> Generator[Session, None, None]:
    """Yield a session and guarantee it is closed (e.g. FastAPI dependency)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables. Phase 1 only; replace with Alembic migrations later."""
    from app import models  # noqa: F401  (registers models on Base.metadata)

    Base.metadata.create_all(bind=engine)

    # Safely migrate new columns if using existing SQLite or PostgreSQL DB
    with engine.connect() as conn:
        for tbl, col, col_type in [
            ("ai_predictions", "is_alternate", "BOOLEAN DEFAULT 0"),
            ("ai_predictions", "cushion_tag", "VARCHAR(50)"),
            ("ai_predictions", "value_line", "FLOAT"),
            ("ai_predictions", "value_probability", "FLOAT"),
            ("ai_predictions", "settlement_status", "VARCHAR(20) DEFAULT 'PENDING'"),
            ("ai_predictions", "settled_at", "DATETIME"),
            ("ai_predictions", "actual_result_context", "VARCHAR(255)"),
            ("entities", "average_xg", "FLOAT"),
            ("entities", "average_possession", "FLOAT"),
            ("entities", "form_rating", "FLOAT DEFAULT 0.0"),
            ("entities", "attack_strength", "FLOAT DEFAULT 0.0"),
            ("entities", "defense_strength", "FLOAT DEFAULT 0.0"),
            ("entities", "avg_xg_for", "FLOAT DEFAULT 0.0"),
            ("entities", "avg_xg_against", "FLOAT DEFAULT 0.0"),
            ("entities", "avg_corners", "FLOAT DEFAULT 0.0"),
            ("entities", "ortg", "FLOAT"),
            ("entities", "drtg", "FLOAT"),
            ("entities", "pace", "FLOAT"),
            ("entities", "set_dominance", "FLOAT"),
            ("entities", "fatigue_index", "FLOAT"),
            ("entities", "form_string", "VARCHAR(20)"),
            ("entities", "goals_scored_per_game", "FLOAT"),
            ("entities", "goals_conceded_per_game", "FLOAT"),
            ("entities", "over_2_5_pct", "FLOAT"),
            ("entities", "clean_sheets", "INTEGER"),
            ("entities", "rolling_xga_per_game", "FLOAT DEFAULT 1.2"),
            ("entities", "rolling_shots_on_target_conceded", "FLOAT DEFAULT 4.0"),
            ("entities", "big_chances_ratio", "FLOAT DEFAULT 1.0"),
            ("entities", "first_half_clean_sheet_pct", "FLOAT DEFAULT 50.0"),
            ("entities", "stalemate_index", "FLOAT DEFAULT 0.0"),
            ("entities", "foul_card_intensity", "FLOAT DEFAULT 2.0"),
            ("entities", "promotion_tax_applied", "BOOLEAN DEFAULT 0"),
            ("entities", "logo_url", "VARCHAR(512)"),
            ("matches", "expected_goals_home", "FLOAT"),
            ("matches", "expected_goals_away", "FLOAT"),
            ("matches", "corner_count", "INTEGER"),
            ("matches", "expected_total_goals", "FLOAT"),
            ("matches", "home_win_prob_baseline", "FLOAT"),
            ("matches", "away_win_prob_baseline", "FLOAT"),
            ("matches", "actual_home_score", "INTEGER"),
            ("matches", "actual_away_score", "INTEGER"),
            ("matches", "prediction_outcome_1x2", "VARCHAR(20)"),
            ("matches", "prediction_exact_score", "BOOLEAN"),
            ("matches", "is_graded", "BOOLEAN DEFAULT 0"),
        ]:
            try:
                conn.execute(text(f"ALTER TABLE {tbl} ADD COLUMN {col} {col_type}"))
                conn.commit()
            except Exception:
                pass

    # Startup purge of mock / corrupted fixtures
    try:
        purge_mock_matches()
    except Exception as p_err:
        logger.debug("Startup mock purge skipped: %s", p_err)


def purge_mock_matches(session: Optional[Session] = None) -> dict[str, int]:
    """Purge mocked fixtures, duplicate Barry Town entries, and matches stuck in LIVE.

    - Deletes all matches associated with Barry Town.
    - Deletes matches with status LIVE whose kickoff time is in the future (> 1 hour ahead).
    - Deletes matches with status LIVE whose kickoff time is ancient past (> 4 hours ago).
    - Removes identical duplicate fixtures (same home, away, date).
    """
    from datetime import datetime, timezone, timedelta
    from app.models import Match, Entity, MatchStatus
    from sqlalchemy import select, or_, delete

    close_on_exit = False
    if session is None:
        session = SessionLocal()
        close_on_exit = True

    deleted_counts = {
        "barry_town_matches": 0,
        "stuck_live_matches": 0,
        "duplicate_fixtures": 0,
        "total_deleted": 0,
    }

    try:
        now_utc = datetime.now(timezone.utc)

        # 1. Purge Barry Town entries
        barry_match_ids = list(
            session.scalars(
                select(Match.id)
                .join(Entity, or_(Match.home_entity_id == Entity.id, Match.away_entity_id == Entity.id))
                .where(Entity.name.ilike("%barry town%"))
            ).all()
        )
        if barry_match_ids:
            session.execute(delete(Match).where(Match.id.in_(barry_match_ids)))
            deleted_counts["barry_town_matches"] = len(barry_match_ids)
            session.commit()

        # 2. Purge matches stuck in LIVE with future kickoff (> 1h from now)
        future_live_cutoff = now_utc + timedelta(hours=1)
        stuck_live_ids = list(
            session.scalars(
                select(Match.id).where(
                    Match.status == MatchStatus.LIVE,
                    Match.kickoff_time > future_live_cutoff,
                )
            ).all()
        )
        # Also clean matches stuck in LIVE for more than 4 hours in the past
        past_live_cutoff = now_utc - timedelta(hours=4)
        past_stuck_ids = list(
            session.scalars(
                select(Match.id).where(
                    Match.status == MatchStatus.LIVE,
                    Match.kickoff_time < past_live_cutoff,
                )
            ).all()
        )
        all_stuck_ids = list(set(stuck_live_ids + past_stuck_ids))
        if all_stuck_ids:
            session.execute(delete(Match).where(Match.id.in_(all_stuck_ids)))
            deleted_counts["stuck_live_matches"] = len(all_stuck_ids)
            session.commit()

        # 3. Purge duplicate fixtures (same home_entity_id, away_entity_id, date)
        all_matches = session.scalars(select(Match).order_by(Match.id.asc())).all()
        seen = set()
        duplicate_ids = []
        for m in all_matches:
            if m.kickoff_time:
                key = (m.home_entity_id, m.away_entity_id, m.kickoff_time.date())
                if key in seen:
                    duplicate_ids.append(m.id)
                else:
                    seen.add(key)
        if duplicate_ids:
            session.execute(delete(Match).where(Match.id.in_(duplicate_ids)))
            deleted_counts["duplicate_fixtures"] = len(duplicate_ids)
            session.commit()

        deleted_counts["total_deleted"] = (
            deleted_counts["barry_town_matches"]
            + deleted_counts["stuck_live_matches"]
            + deleted_counts["duplicate_fixtures"]
        )
        logger.info("Purged mock data from Match table: %s", deleted_counts)

    except Exception as exc:
        session.rollback()
        logger.error("Error purging mock matches: %s", exc)
    finally:
        if close_on_exit:
            session.close()

    return deleted_counts


def purge_all_matches(session: Optional[Session] = None) -> int:
    """Wipe all existing records in the Match table and dependent child rows (odds, predictions).
    Resets the live fixture state completely to allow fresh harvest from official feeds.
    """
    from app.models import AIPrediction, BookmakerOdds, Match, PlayerMatchStats, TeamAdvancedStats
    from sqlalchemy import delete

    close_on_exit = False
    if session is None:
        session = SessionLocal()
        close_on_exit = True

    try:
        session.execute(delete(BookmakerOdds))
        session.execute(delete(AIPrediction))
        session.execute(delete(TeamAdvancedStats))
        session.execute(delete(PlayerMatchStats))
        result = session.execute(delete(Match))
        deleted_count = result.rowcount or 0
        session.commit()
        logger.info("Purged all %d existing records from Match table.", deleted_count)
        return deleted_count
    except Exception as exc:
        session.rollback()
        logger.error("Error purging all matches: %s", exc)
        return 0
    finally:
        if close_on_exit:
            session.close()
