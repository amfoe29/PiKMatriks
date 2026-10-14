"""PikMatrix AI Inference Dashboard Web Server.

Modular FastAPI Application for Live Multi-Sport AI Predictions,
Quantitative Telemetry, and Portfolio Risk Analytics.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except Exception:
        pass

from fastapi import Depends, FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select

from app.auth import get_password_hash
from app.database import SessionLocal, get_db, init_db
from app.models import User
from app.scraper import scrape_daily_fixtures
from app.sync_engine import (
    enrich_matches_with_telemetry,
    grade_finished_predictions,
    run_telemetry_synthesis,
    sync_daily_schedule,
    sync_fixtures_to_db,
)

# Services re-exports for backward compatibility
from app.services.auth_service import get_current_user
from app.services.match_service import (
    build_calendar_days,
    compute_score_distribution,
    compute_win_probabilities,
    enrich_match_for_ui,
    format_consumer_action_title,
    format_kickoff,
    format_tier_action_label,
    generate_match_insight,
    get_league_code,
    get_model_retrain_metadata,
    get_ordinal_suffix,
    get_team_initials,
)

# Modular Routers
from app.routers.api import router as api_router
from app.routers.auth import init_auth_router
from app.routers.dashboard import init_dashboard_router
from app.routers.matches import init_matches_router
from app.routers.profile import init_profile_router
from app.routers.tools import init_tools_router

log = logging.getLogger("pikmatrix_web")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = PROJECT_ROOT / "templates"
STATIC_DIR = PROJECT_ROOT / "static"

if not TEMPLATES_DIR.exists():
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
if not STATIC_DIR.exists():
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "css").mkdir(parents=True, exist_ok=True)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
templates.env.filters["format_kickoff"] = format_kickoff
templates.env.filters["ordinal"] = get_ordinal_suffix
templates.env.globals["build_calendar_days"] = build_calendar_days

# Background scheduler task registry
_background_tasks: set[asyncio.Task] = set()


# --------------------------------------------------------------------------- #
# Background Schedulers: Heavy Loop (12h), Live Loop (60s), Daily Harvest (24h)
# --------------------------------------------------------------------------- #
async def heavy_sync_loop() -> None:
    """Heavy Loop: Runs scrape_daily_fixtures, sync_fixtures_to_db, and enrich_matches_with_telemetry."""
    log.info("Starting Heavy Loop background scheduler (12-hour interval)...")
    await asyncio.sleep(20)
    while True:
        try:
            log.info("[Heavy Loop] Initiating scheduled daily fixtures scrape...")
            fixtures = await scrape_daily_fixtures()

            def _sync_worker():
                with SessionLocal() as session:
                    return sync_fixtures_to_db(fixtures, session)

            sync_res = await asyncio.to_thread(_sync_worker)
            log.info("[Heavy Loop] Daily fixtures sync completed: %s", sync_res)

            with SessionLocal() as session:
                enrich_res = await enrich_matches_with_telemetry(session, max_matches=10)
            log.info("[Heavy Loop] Deep match telemetry enrichment completed: %s", enrich_res)

        except asyncio.CancelledError:
            log.info("[Heavy Loop] Background scheduler cancelled.")
            break
        except Exception as exc:
            log.error("[Heavy Loop] Error in heavy sync loop: %s", exc, exc_info=True)

        await asyncio.sleep(12 * 3600)


async def live_score_loop() -> None:
    """Fast Live-Score Loop: Updates scores and statuses every 60s, grades predictions every 15m."""
    log.info("Starting Fast Live-Score Loop background scheduler (60-second interval)...")
    await asyncio.sleep(2)
    last_grade_time = 0.0
    while True:
        try:
            today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            with SessionLocal() as session:
                sync_res = await sync_daily_schedule(session, today_str)
            log.info("[Live Score Loop] Today's schedule & score sync completed: %s", sync_res)

            now_mono = asyncio.get_event_loop().time()
            if now_mono - last_grade_time >= 900.0:
                try:
                    with SessionLocal() as grade_session:
                        grading_summary = await grade_finished_predictions(grade_session)
                        log.info("[Live Score Loop] Automated prediction grading completed: %s", grading_summary)
                    last_grade_time = now_mono
                except Exception as grade_exc:
                    log.error("[Live Score Loop] Periodic grading error: %s", grade_exc, exc_info=True)

        except asyncio.CancelledError:
            log.info("[Live Score Loop] Background scheduler cancelled.")
            break
        except Exception as exc:
            log.error("[Live Score Loop] Error in live score loop: %s", exc, exc_info=True)

        await asyncio.sleep(60)


light_sync_loop = live_score_loop


async def daily_harvest_loop() -> None:
    """Universal Daily Harvester: Syncs multi-sport schedule for today + 3 days once every 24h."""
    log.info("Starting Daily Harvest background scheduler (24-hour cycle)...")
    await asyncio.sleep(5)
    while True:
        try:
            today_utc = datetime.now(timezone.utc).date()
            dates_to_harvest = [(today_utc + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(4)]
            log.info("[Daily Harvester] Harvesting universal multi-sport schedules for 4 days: %s", dates_to_harvest)

            for d_str in dates_to_harvest:
                with SessionLocal() as session:
                    res = await sync_daily_schedule(session, d_str)
                log.info("[Daily Harvester] Harvest result for %s: %s", d_str, res)

            with SessionLocal() as session:
                synth_res = await run_telemetry_synthesis(session)
            log.info("[Daily Harvester] Telemetry synthesis complete: %s", synth_res)

        except asyncio.CancelledError:
            log.info("[Daily Harvester] Background scheduler cancelled.")
            break
        except Exception as exc:
            log.error("[Daily Harvester] Error in daily harvest loop: %s", exc, exc_info=True)

        now_dt = datetime.now(timezone.utc)
        target_0100 = (now_dt + timedelta(days=1)).replace(hour=1, minute=0, second=0, microsecond=0)
        secs_until_0100 = (target_0100 - now_dt).total_seconds()
        sleep_duration = min(max(3600.0, secs_until_0100), 24 * 3600.0)
        await asyncio.sleep(sleep_duration)


def startup_db_and_seed() -> None:
    """Ensure database tables are initialized and default demo user is created."""
    try:
        init_db()
        with SessionLocal() as session:
            demo_user = session.scalar(select(User).where(User.email == "demo@pikmatrix.com"))
            if not demo_user:
                hashed = get_password_hash("pikmatrix2026")
                session.add(
                    User(
                        email="demo@pikmatrix.com",
                        username="DemoBettor",
                        hashed_password=hashed,
                    )
                )
                session.commit()
                log.info("Initialized default demo user: demo@pikmatrix.com")
    except Exception as e:
        log.warning("Startup database check/seed error: %s", e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager ensuring non-blocking background loops."""
    startup_db_and_seed()
    task_heavy = asyncio.create_task(heavy_sync_loop())
    task_light = asyncio.create_task(light_sync_loop())
    task_harvest = asyncio.create_task(daily_harvest_loop())
    _background_tasks.add(task_heavy)
    _background_tasks.add(task_light)
    _background_tasks.add(task_harvest)
    task_heavy.add_done_callback(_background_tasks.discard)
    task_light.add_done_callback(_background_tasks.discard)
    task_harvest.add_done_callback(_background_tasks.discard)
    log.info("Registered Heavy Loop (12h), Light Loop (60s), and Daily Harvest Loop (24h) background tasks.")
    yield
    for task in list(_background_tasks):
        task.cancel()
    _background_tasks.clear()
    log.info("Background schedulers stopped.")


# --------------------------------------------------------------------------- #
# Application Factory
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="PikMatrix Sports Intelligence",
    description="Live AI-powered multi-sport predictive analytics & telemetry terminal",
    version="2.1.0",
    lifespan=lifespan,
)

# Mount Static Assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Mount Routers
app.include_router(init_auth_router(templates))
app.include_router(init_matches_router(templates))
app.include_router(init_dashboard_router(templates))
app.include_router(init_tools_router(templates))
app.include_router(init_profile_router(templates))
app.include_router(api_router)
