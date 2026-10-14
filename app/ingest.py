"""Ingestion bridge: scraper output -> database (Phase 2).

Flow:
    payloads (live scrape OR ./captures replay)
        -> scraper_test.parse_payloads()         normalised match records
        -> resolve sport / entities / match      in-memory identity maps
        -> bulk INSERT bookmaker_odds            one snapshot per run

Usage (from the project root):
    .\\.venv\\Scripts\\python.exe app/ingest.py --from-captures         # offline replay of ./captures
    .\\.venv\\Scripts\\python.exe app/ingest.py --from-captures DIR     # replay a specific folder
    .\\.venv\\Scripts\\python.exe app/ingest.py --dump                  # live scrape of DEFAULT_URLS
    .\\.venv\\Scripts\\python.exe app/ingest.py <url> <url> ...         # live scrape of given pages
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

# `python app/ingest.py` puts app/ (not the project root) on sys.path, which
# would break `import app...` and `import scraper_test`. Add the root first.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy import distinct, func, insert, inspect, select  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.database import DATABASE_URL, SessionLocal, engine, init_db  # noqa: E402
from app.models import BookmakerOdds, Entity, Match, MatchStatus, Sport  # noqa: E402
from scraper_test import (  # noqa: E402
    DEFAULT_URLS,
    MAX_WAIT_S,
    TARGET_LINE,
    dump_all,
    flatten,
    load_captures,
    parse_payloads,
    scrape,
)

log = logging.getLogger("ingest")

BOOKMAKER_NAME = "SportyBet"
EXTERNAL_ID_SOURCE = "sportybet"  # key under entities.metadata["external_ids"]
INDIVIDUAL_SPORTS = {"tennis"}    # entities.is_individual = True for these
# A fixture already in the DB is the "same" match only if its kickoff is within
# this window. Without it, (competition, home, away) alone would merge
# different games: NBA teams host the same opponent twice a season, and every
# league repeats fixtures season after season.
KICKOFF_MATCH_WINDOW = timedelta(days=3)


@dataclass
class IngestStats:
    records: int = 0
    skipped: int = 0
    sports_created: int = 0
    entities_created: int = 0
    entities_linked: int = 0
    matches_created: int = 0
    matches_rescheduled: int = 0
    matches_unchanged: int = 0
    odds_inserted: int = 0


def as_utc(dt: datetime) -> datetime:
    """SQLite returns naive datetimes; we always store UTC, so re-attach it."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def ensure_schema() -> None:
    """Create missing tables; refuse to run against a pre-competition_name DB."""
    url = str(DATABASE_URL)
    if url.startswith("sqlite:///"):
        db_file = Path(url.removeprefix("sqlite:///"))
        if not db_file.exists():
            log.info("creating new SQLite database at %s", db_file.resolve())

    inspector = inspect(engine)
    if inspector.has_table("matches"):
        columns = {col["name"] for col in inspector.get_columns("matches")}
        if "competition_name" not in columns:
            raise SystemExit(
                "matches table predates the competition_name column. Phase 1 has no "
                "Alembic: delete pikmatrix.db and re-run to recreate the schema."
            )

    init_db()  # create_all() is idempotent: only missing tables are created


# --------------------------------------------------------------------------- #
# In-memory resolvers (one SELECT each at start-up, then dictionary lookups)
# --------------------------------------------------------------------------- #
class SportResolver:
    def __init__(self, session: Session, stats: IngestStats) -> None:
        self.session, self.stats = session, stats
        self.by_slug = {s.slug: s for s in session.scalars(select(Sport))}

    def get(self, slug: str) -> Sport:
        sport = self.by_slug.get(slug)
        if sport is None:
            sport = Sport(name=slug, slug=slug)
            self.session.add(sport)
            self.session.flush()  # assign sport.id now (SessionLocal has autoflush=False)
            self.by_slug[slug] = sport
            self.stats.sports_created += 1
        return sport


class EntityResolver:
    """Identity map for teams/players: external id first, exact name second."""

    def __init__(self, session: Session, stats: IngestStats) -> None:
        self.session, self.stats = session, stats
        self.by_external_id: dict[tuple[int, str], Entity] = {}
        self.by_name: dict[tuple[int, str], Entity] = {}
        for entity in session.scalars(select(Entity)):
            self._index(entity)

    @staticmethod
    def _external_id(entity: Entity) -> Optional[str]:
        return ((entity.entity_metadata or {}).get("external_ids") or {}).get(EXTERNAL_ID_SOURCE)

    def _index(self, entity: Entity) -> None:
        self.by_name[(entity.sport_id, entity.name.casefold())] = entity
        if external_id := self._external_id(entity):
            self.by_external_id[(entity.sport_id, external_id)] = entity

    def resolve(
        self,
        sport: Sport,
        name: str,
        external_id: Optional[str],
        is_national: bool = False,
    ) -> Entity:
        import re
        if not external_id and is_national:
            clean_slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            external_id = f"nat:{clean_slug}"

        entity = self.by_external_id.get((sport.id, external_id)) if external_id else None

        if entity is None:
            entity = self.by_name.get((sport.id, name.casefold()))
            if entity is not None and external_id:
                self._link_external_id(entity, external_id)

        if entity is None:
            metadata: dict[str, Any] = {}
            if external_id:
                metadata["external_ids"] = {EXTERNAL_ID_SOURCE: str(external_id)}
            if is_national or (external_id and str(external_id).startswith("nat:")):
                metadata["is_national_team"] = True
            entity = Entity(
                sport_id=sport.id,
                name=name,
                is_individual=sport.slug in INDIVIDUAL_SPORTS,
                entity_metadata=metadata,
            )
            self.session.add(entity)
            self.session.flush()
            self.stats.entities_created += 1
        else:
            if is_national or (external_id and str(external_id).startswith("nat:")):
                meta = dict(entity.entity_metadata or {})
                if not meta.get("is_national_team"):
                    meta["is_national_team"] = True
                    entity.entity_metadata = meta

        self._index(entity)
        return entity

    def _link_external_id(self, entity: Entity, external_id: str) -> None:
        current = self._external_id(entity)
        if current == str(external_id):
            return
        if current:
            log.warning("%r already linked to %s; feed now says %s - keeping existing link",
                        entity.name, current, external_id)
            return
        metadata = dict(entity.entity_metadata or {})
        ext_ids = dict(metadata.get("external_ids") or {})
        ext_ids[EXTERNAL_ID_SOURCE] = str(external_id)
        metadata["external_ids"] = ext_ids
        # Re-assign a NEW dict: plain JSON columns don't detect in-place mutation.
        entity.entity_metadata = metadata
        self.stats.entities_linked += 1


class MatchResolver:
    """Find-or-create on (sport_id, competition_name, home, away) + kickoff window."""

    def __init__(self, session: Session, stats: IngestStats) -> None:
        self.session, self.stats = session, stats

    def upsert(self, sport: Sport, competition: str, home: Entity, away: Entity,
               kickoff: datetime) -> Match:
        candidates = self.session.scalars(
            select(Match).where(
                Match.sport_id == sport.id,
                Match.competition_name == competition,
                Match.home_entity_id == home.id,
                Match.away_entity_id == away.id,
            )
        )
        for match in candidates:
            stored = as_utc(match.kickoff_time)
            if abs(stored - kickoff) <= KICKOFF_MATCH_WINDOW:
                if stored != kickoff:  # rescheduled - follow the bookmaker
                    match.kickoff_time = kickoff
                    self.stats.matches_rescheduled += 1
                else:
                    self.stats.matches_unchanged += 1
                return match

        match = Match(
            sport_id=sport.id,
            competition_name=competition,
            home_entity_id=home.id,
            away_entity_id=away.id,
            kickoff_time=kickoff,
            status=MatchStatus.SCHEDULED,
        )
        self.session.add(match)
        self.session.flush()  # need match.id for the odds rows
        self.stats.matches_created += 1
        return match


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def ingest(records: list[dict], bookmaker: str = BOOKMAKER_NAME) -> IngestStats:
    """Persist parsed records in ONE transaction (all-or-nothing)."""
    ensure_schema()
    stats = IngestStats(records=len(records))
    # Default fallback timestamp for live runs without per-record timestamps
    default_scraped_at = datetime.now(timezone.utc)
    odds_rows: list[dict] = []

    with SessionLocal() as session, session.begin():  # commits on success, rolls back on error
        sports = SportResolver(session, stats)
        entities = EntityResolver(session, stats)
        matches = MatchResolver(session, stats)

        for record in records:
            kickoff_iso = record.get("kickoff_utc")
            if not (record.get("sport") and record.get("competition") and kickoff_iso):
                stats.skipped += 1
                log.warning("skip %s: missing sport/competition/kickoff", record.get("match"))
                continue

            sport = sports.get(record["sport"])
            is_nat = bool(record.get("is_national"))
            home = entities.resolve(sport, record["home"], record.get("home_id"), is_national=is_nat)
            away = entities.resolve(sport, record["away"], record.get("away_id"), is_national=is_nat)
            match = matches.upsert(
                sport, record["competition"], home, away,
                as_utc(datetime.fromisoformat(kickoff_iso)),
            )

            # Use file modification timestamp if loaded from captures, else default
            record_scraped_at = record.get("scraped_at") or default_scraped_at

            odds_rows.extend(
                {
                    "match_id": match.id,
                    "bookmaker_name": bookmaker,
                    "market_type": line["market_type"],
                    "selection": line["selection"],
                    "line": line["line"],
                    "odds": line["odds"],
                    "scraped_at": record_scraped_at,
                }
                for line in record["odds"]
            )

        # Bulk insert: a single executemany instead of one INSERT per ORM object.
        if odds_rows:
            session.execute(insert(BookmakerOdds), odds_rows)
        stats.odds_inserted = len(odds_rows)

    return stats


ingest_records = ingest


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_summary() -> None:
    with SessionLocal() as session:
        entity_rows = session.execute(
            select(Sport.slug, func.count(Entity.id))
            .join(Entity, Entity.sport_id == Sport.id)
            .group_by(Sport.slug).order_by(Sport.slug)
        ).all()
        match_rows = session.execute(
            select(Sport.slug, Match.competition_name, func.count(Match.id))
            .join(Sport, Sport.id == Match.sport_id)
            .group_by(Sport.slug, Match.competition_name)
            .order_by(Sport.slug, Match.competition_name)
        ).all()
        odds_rows = session.execute(
            select(
                BookmakerOdds.market_type,
                func.count(BookmakerOdds.id),
                func.count(distinct(BookmakerOdds.match_id)),
                func.count(distinct(BookmakerOdds.scraped_at)),
            )
            .group_by(BookmakerOdds.market_type)
            .order_by(BookmakerOdds.market_type)
        ).all()

    print("\n=== DATABASE SUMMARY ===")
    print(f"\nentities ({sum(r[1] for r in entity_rows)} total)")
    for sport, count in entity_rows:
        print(f"  {sport:<12} {count:>6}")
    print(f"\nmatches ({sum(r[2] for r in match_rows)} total)")
    for sport, competition, count in match_rows:
        print(f"  {sport:<12} {competition:<40} {count:>6}")
    print(f"\nbookmaker_odds ({sum(r[1] for r in odds_rows)} total)")
    print(f"  {'market_type':<20} {'rows':>6} {'matches':>8} {'snapshots':>10}")
    for market_type, rows, n_matches, snapshots in odds_rows:
        print(f"  {market_type:<20} {rows:>6} {n_matches:>8} {snapshots:>10}")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="[ingest] %(message)s")
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Scrape (or replay) odds and persist them.")
    parser.add_argument("urls", nargs="*", help="League pages for a live scrape (default: DEFAULT_URLS).")
    parser.add_argument("--from-captures", type=Path, nargs="?", const=PROJECT_ROOT / "captures",
                        metavar="DIR", help="Replay dumped payloads instead of scraping (default: ./captures).")
    parser.add_argument("--dump", action="store_true", help="Live mode: also save raw payloads to ./captures.")
    parser.add_argument("--headed", action="store_true", help="Live mode: show the browser.")
    parser.add_argument("--line", type=float, default=TARGET_LINE, help="Over/Under line to keep.")
    parser.add_argument("--max-wait", type=float, default=MAX_WAIT_S, help="Live mode: seconds per page.")
    parser.add_argument("--sync-international", action="store_true", help="Sync scheduled international break tournaments & matches.")
    args = parser.parse_args()

    if args.sync_international:
        from sync_international import sync_all_international_break_data
        sync_all_international_break_data()
        return 0

    if args.from_captures:
        payloads = load_captures(args.from_captures)
    else:
        payloads_by_page = scrape(args.urls or DEFAULT_URLS, args.headed, args.max_wait)
        if args.dump:
            dump_all(payloads_by_page, PROJECT_ROOT / "captures")
        payloads = flatten(payloads_by_page)

    records = parse_payloads(payloads, args.line)
    if not records:
        log.error("no match records parsed - nothing to ingest")
        return 1

    stats = ingest(records)
    log.info("run stats: %s", ", ".join(f"{f.name}={getattr(stats, f.name)}" for f in fields(stats)))
    print_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
