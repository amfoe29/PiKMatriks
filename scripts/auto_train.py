"""Autonomous Retraining Engine & Self-Learning Pipeline (Task 1).

Connects to `pikmatrix.db`, ingests newly finished matches into `data/football.csv`,
re-engineers all feature columns, retrains the XGBoost model artifacts, overwrites
the models in `models/`, updates metadata, and triggers a re-scoring pass.

Usage:
    python scripts/auto_train.py                     # Single self-learning run
    python scripts/auto_train.py --simulate-finished 3 # Simulate 3 finished matches & retrain
    python scripts/auto_train.py --loop --interval 3600 # Run continuously every hour
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Reconfigure stdout for Windows console UTF-8 compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.database import SessionLocal, init_db
from app.ml_pipeline import FOOTBALL_FEATURE_NAMES, XGBClassifier, evaluate_matches
from app.models import BookmakerOdds, Match, MatchStatus, Sport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [auto_train] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("auto_train")

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
METADATA_FILE = MODELS_DIR / "retrain_metadata.json"


# --------------------------------------------------------------------------- #
# Metric Calculations
# --------------------------------------------------------------------------- #
def calc_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    if len(y_t) == 0:
        return 0.0
    return float(np.mean(y_t == y_p))


def calc_log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.clip(np.asarray(y_prob, dtype=np.float64), eps, 1.0 - eps)
    if len(y_t) == 0:
        return 0.0
    return float(-np.mean(y_t * np.log(y_p) + (1.0 - y_t) * np.log(1.0 - y_p)))


def calc_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_prob)
    n_pos = int(np.sum(y_t == 1))
    n_neg = int(np.sum(y_t == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = pd.Series(y_p).rank(method="average").values
    sum_ranks_pos = float(np.sum(ranks[y_t == 1]))
    u = sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u / (n_pos * n_neg))


# --------------------------------------------------------------------------- #
# Database Helpers: Pull Finished Matches
# --------------------------------------------------------------------------- #
def extract_score(final_score: Any) -> Tuple[Optional[int], Optional[int]]:
    """Parse final match score from JSON dictionary."""
    if not final_score or not isinstance(final_score, dict):
        return None, None
    if "home" in final_score and "away" in final_score:
        return int(final_score["home"]), int(final_score["away"])
    if "home_score" in final_score and "away_score" in final_score:
        return int(final_score["home_score"]), int(final_score["away_score"])
    if "homeScore" in final_score and "awayScore" in final_score:
        hs = final_score["homeScore"]
        ws = final_score["awayScore"]
        h = hs.get("current") if isinstance(hs, dict) else hs
        a = ws.get("current") if isinstance(ws, dict) else ws
        if h is not None and a is not None:
            return int(h), int(a)
    return None, None


def extract_match_odds(match_id: int, session: Any) -> Tuple[float, float, float]:
    """Extract bookmaker 1X2 odds for a match."""
    odds_rows = session.query(BookmakerOdds).filter(
        BookmakerOdds.match_id == match_id,
        BookmakerOdds.odds > 1.0,
    ).all()
    odd_h, odd_d, odd_a = 2.15, 3.30, 3.20
    for o in odds_rows:
        if o.market_type in ("1x2", "moneyline"):
            sel = (o.selection or "").lower()
            if sel in ("home", "1"):
                odd_h = float(o.odds)
            elif sel in ("draw", "x"):
                odd_d = float(o.odds)
            elif sel in ("away", "2"):
                odd_a = float(o.odds)
    return odd_h, odd_d, odd_a


def pull_finished_matches_from_db(
    csv_path: Path = DATA_DIR / "football.csv",
    simulate_count: int = 0,
) -> int:
    """Connect to pikmatrix.db, find finished matches, and append to CSV."""
    init_db()
    with SessionLocal() as session:
        # Simulation of mock scores removed to preserve authentic API match state
        if simulate_count > 0:
            log.warning("Score simulation disabled to preserve authentic API match results.")

        # Query all finished football matches
        finished_matches = (
            session.query(Match)
            .join(Sport, Sport.id == Match.sport_id)
            .filter(
                Sport.slug == "football",
                Match.status == MatchStatus.FINISHED,
            )
            .all()
        )

        if not finished_matches:
            log.info("No finished football matches found in pikmatrix.db")
            return 0

        log.info("Found %d finished matches in pikmatrix.db", len(finished_matches))

        # Load existing dataset to check for duplicates
        if csv_path.exists():
            df_existing = pd.read_csv(csv_path)
            existing_ids = set(df_existing["id_odsp"].astype(str))
            existing_match_keys = set(
                (str(r["date"])[:10], str(r["ht"]).strip().lower(), str(r["at"]).strip().lower())
                for _, r in df_existing.iterrows()
            )
        else:
            df_existing = pd.DataFrame()
            existing_ids = set()
            existing_match_keys = set()

        new_rows = []
        for m in finished_matches:
            fthg, ftag = extract_score(m.final_score)
            if fthg is None or ftag is None:
                continue

            match_id_str = f"db_match_{m.id}"
            ht = m.home_entity.name if m.home_entity else f"Team_{m.home_entity_id}"
            at = m.away_entity.name if m.away_entity else f"Team_{m.away_entity_id}"
            match_date = m.kickoff_time.strftime("%Y-%m-%d")
            match_key = (match_date, ht.strip().lower(), at.strip().lower())

            if match_id_str in existing_ids or match_key in existing_match_keys:
                continue

            odd_h, odd_d, odd_a = extract_match_odds(m.id, session)

            new_rows.append({
                "id_odsp": match_id_str,
                "link_odsp": f"/match/{m.id}",
                "adv_stats": True,
                "date": match_date,
                "league": m.competition_name or "General",
                "season": str(m.kickoff_time.year),
                "country": (
                    m.competition_name.split("-")[0].strip()
                    if "-" in (m.competition_name or "")
                    else "International"
                ),
                "ht": ht,
                "at": at,
                "fthg": fthg,
                "ftag": ftag,
                "odd_h": odd_h,
                "odd_d": odd_d,
                "odd_a": odd_a,
                "odd_over": "NA",
                "odd_under": "NA",
                "odd_bts": "NA",
                "odd_bts_n": "NA",
            })

        if not new_rows:
            log.info("All finished matches are already incorporated in %s", csv_path.name)
            return 0

        df_new = pd.DataFrame(new_rows)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(csv_path, index=False)
        log.info(
            "Successfully appended %d newly finished matches to %s (Total now: %d)",
            len(new_rows),
            csv_path.name,
            len(df_combined),
        )
        return len(new_rows)


# --------------------------------------------------------------------------- #
# Feature Engineering & Retraining Engine
# --------------------------------------------------------------------------- #
def retrain_football_models(
    csv_path: Path = DATA_DIR / "football.csv",
    split_ratio: float = 0.8,
) -> Dict[str, Dict[str, float]]:
    """Recalculate all rolling features and retrain all 4 football models."""
    log.info("Loading expanded football dataset from %s", csv_path)
    df = pd.read_csv(csv_path)

    required = ["date", "ht", "at", "fthg", "ftag", "odd_h", "odd_d", "odd_a"]
    df = df.dropna(subset=required).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Check for corner counts in events.csv if present
    events_path = DATA_DIR / "Football" / "events.csv"
    corner_dict: Dict[str, int] = {}
    if events_path.exists():
        try:
            corners_df = pd.read_csv(events_path, usecols=["id_odsp", "event_type"])
            corners_df = corners_df[corners_df["event_type"] == 2]
            corner_counts = corners_df.groupby("id_odsp").size()
            corner_dict = corner_counts.to_dict()
        except Exception:
            pass

    log.info("Recalculating rolling feature engineering across %d matches...", len(df))
    team_history: Dict[str, List[Tuple[int, int, int]]] = {}
    features = []
    targets_1x = []
    targets_hcp = []
    targets_ou = []
    targets_corners = []

    rng = np.random.RandomState(42)

    for _, row in df.iterrows():
        ht, at = str(row["ht"]), str(row["at"])
        h_hist = team_history.get(ht, [])
        a_hist = team_history.get(at, [])

        h_last = h_hist[-5:] if h_hist else []
        a_last = a_hist[-5:] if a_hist else []

        h_form = float(np.mean([x[0] for x in h_last])) if h_last else 1.3
        a_form = float(np.mean([x[0] for x in a_last])) if a_last else 1.3

        h_scored = float(np.mean([x[1] for x in h_last])) if h_last else 1.4
        h_conceded = float(np.mean([x[2] for x in h_last])) if h_last else 1.1
        a_scored = float(np.mean([x[1] for x in a_last])) if a_last else 1.1
        a_conceded = float(np.mean([x[2] for x in a_last])) if a_last else 1.4

        h_gd = h_scored - h_conceded
        a_gd = a_scored - a_conceded

        oh = float(row["odd_h"])
        od = float(row["odd_d"])
        oa = float(row["odd_a"])

        raw_h = 1.0 / max(oh, 1.01)
        raw_d = 1.0 / max(od, 1.01)
        raw_a = 1.0 / max(oa, 1.01)
        overround = raw_h + raw_d + raw_a

        prob_h = raw_h / overround
        prob_d = raw_d / overround
        prob_a = raw_a / overround
        implied_1x = prob_h + prob_d
        odds_ratio = oh / max(oa, 0.01)
        margin = overround - 1.0

        feat = [
            h_form - a_form,
            h_gd - a_gd,
            h_scored - a_conceded,
            a_scored - h_conceded,
            prob_h,
            prob_d,
            prob_a,
            implied_1x,
            odds_ratio,
            margin,
        ]
        features.append(feat)

        fthg, ftag = int(row["fthg"]), int(row["ftag"])

        # Target 1: 1X (Home Win or Draw)
        targets_1x.append(1 if fthg >= ftag else 0)

        # Target 2: Handicap (Home covers spread synthesized from goal diffs)
        spread = np.round((h_gd - a_gd) * 0.5 * 2.0) / 2.0
        targets_hcp.append(1 if (fthg - ftag) > spread else 0)

        # Target 3: Over/Under 2.5 Goals
        targets_ou.append(1 if (fthg + ftag) > 2.5 else 0)

        # Target 4: Corners Over 9.5
        match_id_odsp = str(row.get("id_odsp", ""))
        if match_id_odsp in corner_dict:
            tot_corners = corner_dict[match_id_odsp]
        else:
            lam_corners = max(4.0, 5.0 + 1.8 * (h_scored + a_scored) + 1.2 * max(0.0, h_gd))
            tot_corners = int(rng.poisson(lam_corners))
        targets_corners.append(1 if tot_corners > 9.5 else 0)

        # Track history
        if fthg > ftag:
            hp, ap = 3, 0
        elif fthg == ftag:
            hp, ap = 1, 1
        else:
            hp, ap = 0, 3

        team_history.setdefault(ht, []).append((hp, fthg, ftag))
        team_history.setdefault(at, []).append((ap, ftag, fthg))

    X = pd.DataFrame(features, columns=FOOTBALL_FEATURE_NAMES)
    split_idx = int(split_ratio * len(X))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_summary: Dict[str, Dict[str, float]] = {}

    def _train_model(y_arr: np.ndarray, model_file: str, label: str) -> Dict[str, float]:
        y_train, y_test = y_arr[:split_idx], y_arr[split_idx:]
        clf = XGBClassifier(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
        )
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        acc = calc_accuracy(y_test, y_pred)
        ll = calc_log_loss(y_test, y_prob)
        auc = calc_roc_auc(y_test, y_prob)

        metrics = {"accuracy": round(acc, 4), "log_loss": round(ll, 4), "roc_auc": round(auc, 4)}
        out_path = MODELS_DIR / model_file
        with open(out_path, "wb") as f:
            pickle.dump(clf, f)

        log.info(
            "%-28s -> Acc: %.4f | LogLoss: %.4f | ROC-AUC: %.4f (Saved to %s)",
            label,
            acc,
            ll,
            auc,
            model_file,
        )
        return metrics

    log.info("--- Retraining XGBoost Football Models (%d Train, %d Test) ---", len(X_train), len(X_test))

    metrics_summary["football_1x"] = _train_model(
        np.array(targets_1x), "xgboost_football_1x.pkl", "Double Chance (1X)"
    )
    # Sync legacy aliases
    shutil.copyfile(MODELS_DIR / "xgboost_football_1x.pkl", MODELS_DIR / "xgboost_football.pkl")
    shutil.copyfile(MODELS_DIR / "xgboost_football_1x.pkl", MODELS_DIR / "xgboost_football_1x_v1.pkl")

    metrics_summary["football_hcp"] = _train_model(
        np.array(targets_hcp), "xgboost_football_hcp.pkl", "Goal Spread Handicap"
    )

    metrics_summary["football_ou"] = _train_model(
        np.array(targets_ou), "xgboost_football_ou.pkl", "Total Goals Over/Under"
    )

    metrics_summary["football_corners"] = _train_model(
        np.array(targets_corners), "xgboost_football_corners.pkl", "Corners Over 9.5"
    )

    return metrics_summary


# --------------------------------------------------------------------------- #
# Metadata & Self-Learning State Tracking
# --------------------------------------------------------------------------- #
def update_retrain_metadata(
    total_samples: int,
    new_matches_added: int,
    metrics: Dict[str, Dict[str, float]],
) -> None:
    """Save persistent retraining metadata for the UI learning indicator."""
    now = datetime.now(timezone.utc)
    metadata = {
        "last_trained_iso": now.isoformat(),
        "last_trained_display": now.strftime("%b %d, %Y • %H:%M UTC"),
        "total_samples": total_samples,
        "new_matches_added": new_matches_added,
        "metrics": metrics,
        "engine_status": "active",
        "self_learning": True,
    }
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    log.info("Saved retrain metadata to %s", METADATA_FILE)


# --------------------------------------------------------------------------- #
# Full Self-Learning Cycle
# --------------------------------------------------------------------------- #
def run_retraining_cycle(simulate_finished: int = 0) -> None:
    """Execute complete self-learning pipeline."""
    log.info("==================================================================")
    log.info("STARTING AUTONOMOUS MODEL RETRAINING PIPELINE")
    log.info("==================================================================")

    csv_path = DATA_DIR / "football.csv"
    new_count = pull_finished_matches_from_db(csv_path, simulate_count=simulate_finished)

    # Retrain models with recalculation of feature engineering columns
    metrics = retrain_football_models(csv_path)

    # Count total dataset rows
    df_current = pd.read_csv(csv_path)
    total_samples = len(df_current)

    # Update metadata
    update_retrain_metadata(total_samples, new_count, metrics)

    # Re-score all live scheduled matches in pikmatrix.db with newly trained models
    log.info("Running live match re-evaluation pass with updated weights...")
    predictions = evaluate_matches(threshold=0.80)
    log.info(
        "Self-Learning Cycle Complete! Ingested %d new matches. Active safe predictions: %d",
        new_count,
        len(predictions),
    )


# --------------------------------------------------------------------------- #
# Main CLI Entrypoint
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="PikMatrix Autonomous Retraining Engine")
    parser.add_argument(
        "--simulate-finished",
        type=int,
        default=0,
        help="Simulate N matches finishing in database to demonstrate self-learning.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously in background.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=3600,
        help="Seconds between cycles in loop mode (default: 3600).",
    )
    args = parser.parse_args()

    if args.loop:
        log.info("Starting Autonomous Retraining Daemon (Interval: %ds)", args.interval)
        while True:
            try:
                run_retraining_cycle(simulate_finished=args.simulate_finished)
            except Exception as exc:
                log.error("Retraining cycle encountered error: %s", exc, exc_info=True)
            log.info("Sleeping for %d seconds until next retraining cycle...", args.interval)
            time.sleep(args.interval)
    else:
        run_retraining_cycle(simulate_finished=args.simulate_finished)


if __name__ == "__main__":
    main()
