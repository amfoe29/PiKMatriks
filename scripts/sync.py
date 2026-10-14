#!/usr/bin/env python3
"""PikMatrix Daily Fixture Scraper & Database Ingestion CLI (scripts/sync.py).

Fetches authentic real-time match fixtures from Sofascore (with live open sports API fallback),
purges stale synthetic/dummy seed fixtures, and synchronizes real matches with calibrated
AI predictions into pikmatrix.db.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.database import SessionLocal
from app.sofascore import (
    fetch_daily_fixtures,
    purge_all_fixtures,
    sync_all_dates,
    sync_daily_fixtures,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("pikmatrix_sync")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync real-world daily match fixtures and purge mock seed data."
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date in YYYY-MM-DD format (e.g. 2026-10-04).",
    )
    parser.add_argument(
        "--all-dates",
        action="store_true",
        help="Sync yesterday, today, and tomorrow fixtures.",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="Purge all old dummy/seed matches from pikmatrix.db prior to ingestion.",
    )

    args = parser.parse_args()

    now_utc = datetime.now(timezone.utc)
    current_dt = now_utc.date()
    yesterday_str = (current_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    today_str = current_dt.strftime("%Y-%m-%d")
    tomorrow_str = (current_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    print("\n" + "=" * 65)
    print("  PIKMATRIX REAL-TIME FIXTURE SYNCHRONIZATION")
    print("=" * 65)
    print(f"Current UTC System Time: {now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Yesterday: {yesterday_str} | Today: {today_str} | Tomorrow: {tomorrow_str}")
    print("=" * 65)

    with SessionLocal() as session:
        # Default behavior (no args specified): purge old seed matches and sync yesterday, today, tomorrow
        if not args.date and not args.all_dates and not args.purge:
            args.all_dates = True
            args.purge = True

        if args.purge and not args.all_dates and not args.date:
            print("\nPurging all legacy/seed matches and associated predictions...")
            count = purge_all_fixtures(session)
            print(f"✅ Successfully purged {count} legacy/seed matches from pikmatrix.db.")
            return 0

        if args.all_dates:
            target_dates = [yesterday_str, today_str, tomorrow_str]
            print(f"\n[1/3] Purging legacy seed data and synchronizing 3 dates: {target_dates}...")
            res = sync_all_dates(dates=target_dates, session=session, purge_first=args.purge)

            print("\n" + "-" * 65)
            print("  SYNCHRONIZATION REPORT")
            print("-" * 65)
            print(f"  • Old Seed Matches Purged:       {res.get('purged_old_matches', 0)}")
            print(f"  • Total Real Matches Synced:     {res.get('total_matches_synced', 0)}")
            print(f"  • Settled Past Matches:          {res.get('settled_matches', 0)}")
            print(f"  • AI Predictions Evaluated:      {res.get('predictions_evaluated', 0)}")
            print("-" * 65)

            for d, details in res.get("details_by_date", {}).items():
                print(f"  📅 Date: {d} -> Total: {details.get('count', 0)} matches")
                for sp, count in details.get("by_sport", {}).items():
                    print(f"      - {sp.capitalize()}: {count} matches")

            print("=" * 65)
            print("✅ Real-world sync complete for yesterday, today, and tomorrow!\n")
            return 0

        elif args.date:
            target_d = args.date.strip()[:10]
            print(f"\nSynchronizing real matches for date: {target_d} (Purge: {args.purge})...")
            res = sync_daily_fixtures(
                target_date=target_d,
                session=session,
                purge_all=args.purge,
            )

            print("\n" + "-" * 65)
            print(f"  DATE SYNC REPORT: {target_d}")
            print("-" * 65)
            print(f"  • Matches Purged:                {res.get('purged_matches', 0)}")
            print(f"  • Real Matches Synced:           {res.get('synced_matches', 0)}")
            for sp, count in res.get("sports_synced", {}).items():
                print(f"      - {sp.capitalize()}: {count} matches")
            print(f"  • AI Predictions Evaluated:      {res.get('predictions_evaluated', 0)}")
            print("-" * 65)
            print(f"✅ Sync complete for {target_d}!\n")
            return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
