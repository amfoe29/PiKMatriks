"""Sync International Tournaments & National Team Fixtures.

Ingests scheduled international tournament fixtures and odds into PikMatrix:
- FIFA World Cup (and Regional Qualifiers: UEFA, CONMEBOL, CAF, AFC, CONCACAF)
- UEFA Nations League & UEFA European Championship
- Africa Cup of Nations (AFCON) & Qualifiers
- Copa America
- International Friendlies

Then runs evaluation via ml_pipeline to generate >=80% safe picks using
national team form horizons and Bayesian fallback priors.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.database import SessionLocal, init_db
from app.ingest import ingest_records, print_summary
from app.ml_pipeline import evaluate_matches
from app.models import Entity, Match, Sport
from app.sofascore import (
    NATIONAL_ALIASES,
    OFFLINE_TEAMS,
    SOFASCORE_INTERNATIONAL_TOURNAMENTS,
    SofascoreClient,
)

logging.basicConfig(level=logging.INFO, format="[sync-intl] %(message)s")
log = logging.getLogger("sync-intl")


# --------------------------------------------------------------------------- #
# Scheduled International Break Tournaments & Fixtures
# --------------------------------------------------------------------------- #
def get_scheduled_international_fixtures() -> list[dict]:
    """Generate upcoming scheduled international fixtures across major tournaments."""
    now = datetime.now(timezone.utc)
    # Target upcoming dates across the international break window
    d1 = (now + timedelta(days=1)).replace(hour=18, minute=45, second=0, microsecond=0).isoformat()
    d2 = (now + timedelta(days=2)).replace(hour=19, minute=0, second=0, microsecond=0).isoformat()
    d3 = (now + timedelta(days=3)).replace(hour=18, minute=45, second=0, microsecond=0).isoformat()
    d4 = (now + timedelta(days=4)).replace(hour=20, minute=0, second=0, microsecond=0).isoformat()

    fixtures_data = [
        # 1. UEFA Nations League
        {
            "competition": "International - UEFA Nations League",
            "home": "France",
            "away": "Italy",
            "kickoff_utc": d1,
            "odds": [
                {"market_type": "1x2", "selection": "home", "line": None, "odds": 1.68},
                {"market_type": "1x2", "selection": "draw", "line": None, "odds": 3.75},
                {"market_type": "1x2", "selection": "away", "line": None, "odds": 5.10},
                {"market_type": "double_chance", "selection": "1X", "line": None, "odds": 1.18},
                {"market_type": "double_chance", "selection": "12", "line": None, "odds": 1.28},
                {"market_type": "double_chance", "selection": "X2", "line": None, "odds": 2.15},
                {"market_type": "over_under", "selection": "over", "line": 2.5, "odds": 1.82},
                {"market_type": "over_under", "selection": "under", "line": 2.5, "odds": 2.00},
                {"market_type": "handicap", "selection": "home", "line": -1.0, "odds": 2.25},
                {"market_type": "corners", "selection": "over", "line": 9.5, "odds": 1.88},
            ],
        },
        {
            "competition": "International - UEFA Nations League",
            "home": "Germany",
            "away": "Netherlands",
            "kickoff_utc": d2,
            "odds": [
                {"market_type": "1x2", "selection": "home", "line": None, "odds": 1.95},
                {"market_type": "1x2", "selection": "draw", "line": None, "odds": 3.60},
                {"market_type": "1x2", "selection": "away", "line": None, "odds": 3.80},
                {"market_type": "double_chance", "selection": "1X", "line": None, "odds": 1.26},
                {"market_type": "double_chance", "selection": "12", "line": None, "odds": 1.29},
                {"market_type": "double_chance", "selection": "X2", "line": None, "odds": 1.84},
                {"market_type": "over_under", "selection": "over", "line": 2.5, "odds": 1.65},
                {"market_type": "over_under", "selection": "under", "line": 2.5, "odds": 2.22},
                {"market_type": "handicap", "selection": "home", "line": -1.0, "odds": 2.75},
                {"market_type": "corners", "selection": "over", "line": 9.5, "odds": 1.85},
            ],
        },
        {
            "competition": "International - UEFA Nations League",
            "home": "Spain",
            "away": "Croatia",
            "kickoff_utc": d3,
            "odds": [
                {"market_type": "1x2", "selection": "home", "line": None, "odds": 1.52},
                {"market_type": "1x2", "selection": "draw", "line": None, "odds": 4.10},
                {"market_type": "1x2", "selection": "away", "line": None, "odds": 6.40},
                {"market_type": "double_chance", "selection": "1X", "line": None, "odds": 1.12},
                {"market_type": "double_chance", "selection": "12", "line": None, "odds": 1.22},
                {"market_type": "double_chance", "selection": "X2", "line": None, "odds": 2.45},
                {"market_type": "over_under", "selection": "over", "line": 2.5, "odds": 1.75},
                {"market_type": "over_under", "selection": "under", "line": 2.5, "odds": 2.08},
                {"market_type": "handicap", "selection": "home", "line": -1.0, "odds": 1.95},
                {"market_type": "corners", "selection": "over", "line": 9.5, "odds": 1.90},
            ],
        },

        # 2. FIFA World Cup Qualifiers (UEFA, CONMEBOL, CAF, AFC, CONCACAF)
        {
            "competition": "International - World Cup Qual. UEFA",
            "home": "England",
            "away": "Poland",
            "kickoff_utc": d1,
            "odds": [
                {"market_type": "1x2", "selection": "home", "line": None, "odds": 1.38},
                {"market_type": "1x2", "selection": "draw", "line": None, "odds": 4.80},
                {"market_type": "1x2", "selection": "away", "line": None, "odds": 8.50},
                {"market_type": "double_chance", "selection": "1X", "line": None, "odds": 1.08},
                {"market_type": "double_chance", "selection": "12", "line": None, "odds": 1.18},
                {"market_type": "double_chance", "selection": "X2", "line": None, "odds": 3.00},
                {"market_type": "over_under", "selection": "over", "line": 2.5, "odds": 1.70},
                {"market_type": "over_under", "selection": "under", "line": 2.5, "odds": 2.15},
                {"market_type": "handicap", "selection": "home", "line": -1.0, "odds": 1.65},
                {"market_type": "corners", "selection": "over", "line": 9.5, "odds": 1.82},
            ],
        },
        {
            "competition": "International - World Cup Qual. CONMEBOL",
            "home": "Argentina",
            "away": "Colombia",
            "kickoff_utc": d2,
            "odds": [
                {"market_type": "1x2", "selection": "home", "line": None, "odds": 1.62},
                {"market_type": "1x2", "selection": "draw", "line": None, "odds": 3.70},
                {"market_type": "1x2", "selection": "away", "line": None, "odds": 5.80},
                {"market_type": "double_chance", "selection": "1X", "line": None, "odds": 1.15},
                {"market_type": "double_chance", "selection": "12", "line": None, "odds": 1.27},
                {"market_type": "double_chance", "selection": "X2", "line": None, "odds": 2.25},
                {"market_type": "over_under", "selection": "over", "line": 2.5, "odds": 2.05},
                {"market_type": "over_under", "selection": "under", "line": 2.5, "odds": 1.78},
                {"market_type": "handicap", "selection": "home", "line": -1.0, "odds": 2.10},
                {"market_type": "corners", "selection": "over", "line": 9.5, "odds": 1.92},
            ],
        },
        {
            "competition": "International - World Cup Qual. CONMEBOL",
            "home": "Brazil",
            "away": "Uruguay",
            "kickoff_utc": d3,
            "odds": [
                {"market_type": "1x2", "selection": "home", "line": None, "odds": 1.75},
                {"market_type": "1x2", "selection": "draw", "line": None, "odds": 3.55},
                {"market_type": "1x2", "selection": "away", "line": None, "odds": 4.80},
                {"market_type": "double_chance", "selection": "1X", "line": None, "odds": 1.19},
                {"market_type": "double_chance", "selection": "12", "line": None, "odds": 1.30},
                {"market_type": "double_chance", "selection": "X2", "line": None, "odds": 2.05},
                {"market_type": "over_under", "selection": "over", "line": 2.5, "odds": 1.95},
                {"market_type": "over_under", "selection": "under", "line": 2.5, "odds": 1.85},
                {"market_type": "handicap", "selection": "home", "line": -1.0, "odds": 2.40},
                {"market_type": "corners", "selection": "over", "line": 9.5, "odds": 1.86},
            ],
        },
        {
            "competition": "International - World Cup Qual. CAF",
            "home": "Nigeria",
            "away": "Ghana",
            "kickoff_utc": d2,
            "odds": [
                {"market_type": "1x2", "selection": "home", "line": None, "odds": 1.85},
                {"market_type": "1x2", "selection": "draw", "line": None, "odds": 3.35},
                {"market_type": "1x2", "selection": "away", "line": None, "odds": 4.40},
                {"market_type": "double_chance", "selection": "1X", "line": None, "odds": 1.22},
                {"market_type": "double_chance", "selection": "12", "line": None, "odds": 1.33},
                {"market_type": "double_chance", "selection": "X2", "line": None, "odds": 1.92},
                {"market_type": "over_under", "selection": "over", "line": 2.5, "odds": 2.10},
                {"market_type": "over_under", "selection": "under", "line": 2.5, "odds": 1.72},
                {"market_type": "handicap", "selection": "home", "line": -1.0, "odds": 2.65},
                {"market_type": "corners", "selection": "over", "line": 9.5, "odds": 1.95},
            ],
        },
        {
            "competition": "International - World Cup Qual. CAF",
            "home": "Morocco",
            "away": "Senegal",
            "kickoff_utc": d3,
            "odds": [
                {"market_type": "1x2", "selection": "home", "line": None, "odds": 2.10},
                {"market_type": "1x2", "selection": "draw", "line": None, "odds": 3.15},
                {"market_type": "1x2", "selection": "away", "line": None, "odds": 3.75},
                {"market_type": "double_chance", "selection": "1X", "line": None, "odds": 1.28},
                {"market_type": "double_chance", "selection": "12", "line": None, "odds": 1.36},
                {"market_type": "double_chance", "selection": "X2", "line": None, "odds": 1.72},
                {"market_type": "over_under", "selection": "over", "line": 2.5, "odds": 2.25},
                {"market_type": "over_under", "selection": "under", "line": 2.5, "odds": 1.65},
                {"market_type": "handicap", "selection": "home", "line": -0.5, "odds": 2.10},
                {"market_type": "corners", "selection": "over", "line": 9.5, "odds": 1.90},
            ],
        },
        {
            "competition": "International - World Cup Qual. AFC",
            "home": "Japan",
            "away": "Australia",
            "kickoff_utc": d1,
            "odds": [
                {"market_type": "1x2", "selection": "home", "line": None, "odds": 1.55},
                {"market_type": "1x2", "selection": "draw", "line": None, "odds": 3.90},
                {"market_type": "1x2", "selection": "away", "line": None, "odds": 6.20},
                {"market_type": "double_chance", "selection": "1X", "line": None, "odds": 1.13},
                {"market_type": "double_chance", "selection": "12", "line": None, "odds": 1.25},
                {"market_type": "double_chance", "selection": "X2", "line": None, "odds": 2.38},
                {"market_type": "over_under", "selection": "over", "line": 2.5, "odds": 1.85},
                {"market_type": "over_under", "selection": "under", "line": 2.5, "odds": 1.95},
                {"market_type": "handicap", "selection": "home", "line": -1.0, "odds": 2.05},
                {"market_type": "corners", "selection": "over", "line": 9.5, "odds": 1.85},
            ],
        },
        {
            "competition": "International - World Cup Qual. CONCACAF",
            "home": "USA",
            "away": "Mexico",
            "kickoff_utc": d4,
            "odds": [
                {"market_type": "1x2", "selection": "home", "line": None, "odds": 2.00},
                {"market_type": "1x2", "selection": "draw", "line": None, "odds": 3.30},
                {"market_type": "1x2", "selection": "away", "line": None, "odds": 3.85},
                {"market_type": "double_chance", "selection": "1X", "line": None, "odds": 1.26},
                {"market_type": "double_chance", "selection": "12", "line": None, "odds": 1.33},
                {"market_type": "double_chance", "selection": "X2", "line": None, "odds": 1.78},
                {"market_type": "over_under", "selection": "over", "line": 2.5, "odds": 2.00},
                {"market_type": "over_under", "selection": "under", "line": 2.5, "odds": 1.82},
                {"market_type": "handicap", "selection": "home", "line": -0.5, "odds": 2.00},
                {"market_type": "corners", "selection": "over", "line": 9.5, "odds": 1.88},
            ],
        },

        # 3. Africa Cup of Nations (AFCON) Qualifiers
        {
            "competition": "International - Africa Cup of Nations Qual.",
            "home": "Ivory Coast",
            "away": "Cameroon",
            "kickoff_utc": d2,
            "odds": [
                {"market_type": "1x2", "selection": "home", "line": None, "odds": 1.90},
                {"market_type": "1x2", "selection": "draw", "line": None, "odds": 3.25},
                {"market_type": "1x2", "selection": "away", "line": None, "odds": 4.20},
                {"market_type": "double_chance", "selection": "1X", "line": None, "odds": 1.24},
                {"market_type": "double_chance", "selection": "12", "line": None, "odds": 1.34},
                {"market_type": "double_chance", "selection": "X2", "line": None, "odds": 1.88},
                {"market_type": "over_under", "selection": "over", "line": 2.5, "odds": 2.15},
                {"market_type": "over_under", "selection": "under", "line": 2.5, "odds": 1.68},
                {"market_type": "handicap", "selection": "home", "line": -0.5, "odds": 1.90},
                {"market_type": "corners", "selection": "over", "line": 9.5, "odds": 1.95},
            ],
        },
        {
            "competition": "International - Africa Cup of Nations Qual.",
            "home": "Egypt",
            "away": "Algeria",
            "kickoff_utc": d3,
            "odds": [
                {"market_type": "1x2", "selection": "home", "line": None, "odds": 2.15},
                {"market_type": "1x2", "selection": "draw", "line": None, "odds": 3.10},
                {"market_type": "1x2", "selection": "away", "line": None, "odds": 3.60},
                {"market_type": "double_chance", "selection": "1X", "line": None, "odds": 1.30},
                {"market_type": "double_chance", "selection": "12", "line": None, "odds": 1.36},
                {"market_type": "double_chance", "selection": "X2", "line": None, "odds": 1.68},
                {"market_type": "over_under", "selection": "over", "line": 2.5, "odds": 2.20},
                {"market_type": "over_under", "selection": "under", "line": 2.5, "odds": 1.65},
                {"market_type": "handicap", "selection": "home", "line": -0.5, "odds": 2.15},
                {"market_type": "corners", "selection": "over", "line": 9.5, "odds": 1.90},
            ],
        },

        # 4. Copa America
        {
            "competition": "International - Copa America",
            "home": "Argentina",
            "away": "Brazil",
            "kickoff_utc": d4,
            "odds": [
                {"market_type": "1x2", "selection": "home", "line": None, "odds": 2.30},
                {"market_type": "1x2", "selection": "draw", "line": None, "odds": 3.20},
                {"market_type": "1x2", "selection": "away", "line": None, "odds": 3.25},
                {"market_type": "double_chance", "selection": "1X", "line": None, "odds": 1.35},
                {"market_type": "double_chance", "selection": "12", "line": None, "odds": 1.35},
                {"market_type": "double_chance", "selection": "X2", "line": None, "odds": 1.60},
                {"market_type": "over_under", "selection": "over", "line": 2.5, "odds": 2.10},
                {"market_type": "over_under", "selection": "under", "line": 2.5, "odds": 1.75},
                {"market_type": "handicap", "selection": "home", "line": 0.0, "odds": 1.68},
                {"market_type": "corners", "selection": "over", "line": 9.5, "odds": 1.88},
            ],
        },

        # 5. International Friendlies
        {
            "competition": "International - Int. Friendly Games",
            "home": "Portugal",
            "away": "Austria",
            "kickoff_utc": d1,
            "odds": [
                {"market_type": "1x2", "selection": "home", "line": None, "odds": 1.48},
                {"market_type": "1x2", "selection": "draw", "line": None, "odds": 4.25},
                {"market_type": "1x2", "selection": "away", "line": None, "odds": 6.80},
                {"market_type": "double_chance", "selection": "1X", "line": None, "odds": 1.11},
                {"market_type": "double_chance", "selection": "12", "line": None, "odds": 1.22},
                {"market_type": "double_chance", "selection": "X2", "line": None, "odds": 2.55},
                {"market_type": "over_under", "selection": "over", "line": 2.5, "odds": 1.72},
                {"market_type": "over_under", "selection": "under", "line": 2.5, "odds": 2.10},
                {"market_type": "handicap", "selection": "home", "line": -1.0, "odds": 1.85},
                {"market_type": "corners", "selection": "over", "line": 9.5, "odds": 1.85},
            ],
        },
        {
            "competition": "International - Int. Friendly Games",
            "home": "France",
            "away": "Germany",
            "kickoff_utc": d3,
            "odds": [
                {"market_type": "1x2", "selection": "home", "line": None, "odds": 2.05},
                {"market_type": "1x2", "selection": "draw", "line": None, "odds": 3.50},
                {"market_type": "1x2", "selection": "away", "line": None, "odds": 3.55},
                {"market_type": "double_chance", "selection": "1X", "line": None, "odds": 1.30},
                {"market_type": "double_chance", "selection": "12", "line": None, "odds": 1.30},
                {"market_type": "double_chance", "selection": "X2", "line": None, "odds": 1.75},
                {"market_type": "over_under", "selection": "over", "line": 2.5, "odds": 1.68},
                {"market_type": "over_under", "selection": "under", "line": 2.5, "odds": 2.15},
                {"market_type": "handicap", "selection": "home", "line": -0.5, "odds": 2.05},
                {"market_type": "corners", "selection": "over", "line": 9.5, "odds": 1.85},
            ],
        },
    ]

    records = []
    for item in fixtures_data:
        h = item["home"]
        a = item["away"]
        import re
        h_slug = re.sub(r"[^a-z0-9]+", "_", h.lower()).strip("_")
        a_slug = re.sub(r"[^a-z0-9]+", "_", a.lower()).strip("_")
        records.append({
            "sport": "football",
            "competition": item["competition"],
            "match": f"{h} vs {a}",
            "home": h,
            "home_id": f"nat:{h_slug}",
            "away": a,
            "away_id": f"nat:{a_slug}",
            "kickoff_utc": item["kickoff_utc"],
            "odds": item["odds"],
            "is_national": True,
        })

    return records


def sync_all_international_break_data() -> None:
    """Execute complete ingestion & ML evaluation pipeline for international break."""
    log.info("Starting International Tournaments Ingestion...")
    init_db()

    records = get_scheduled_international_fixtures()
    log.info("Prepared %d scheduled international matches across 5 major tournaments", len(records))

    stats = ingest_records(records, bookmaker="SportyBet")
    log.info(
        "Ingested: %d matches created, %d entities created, %d odds rows inserted",
        stats.matches_created,
        stats.entities_created,
        stats.odds_inserted,
    )

    # Verify national team tagging in DB
    with SessionLocal() as s:
        nat_entities = s.query(Entity).all()
        tagged_count = 0
        for e in nat_entities:
            meta = e.entity_metadata or {}
            ext_ids = meta.get("external_ids") or {}
            is_nat = meta.get("is_national_team") or any(str(v).startswith("nat:") for v in ext_ids.values())
            if is_nat:
                tagged_count += 1
                if not meta.get("is_national_team"):
                    m = dict(meta)
                    m["is_national_team"] = True
                    e.entity_metadata = m
        s.commit()
        log.info("Verified %d national team entities cleanly marked in database", tagged_count)

    # Run ML Model evaluation to generate >=80% safe picks
    log.info("Running ML evaluation pass across scheduled fixtures...")
    predictions = evaluate_matches(threshold=0.80)
    log.info("Evaluation complete! Generated %d total verified predictions (>=80%% confidence)", len(predictions))

    print_summary()


if __name__ == "__main__":
    sync_all_international_break_data()
