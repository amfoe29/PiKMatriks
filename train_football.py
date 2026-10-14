"""Standalone Training Engine for Football (Soccer) Models (Task 1).

Trains and saves 4 sport-and-market-specific XGBoost models:
1. `xgboost_football_1x.pkl`      (Target: Double Chance 1X - Home Win or Draw)
2. `xgboost_football_hcp.pkl`     (Target: Goal Handicap Cover against spread)
3. `xgboost_football_ou.pkl`      (Target: Total Goals Over 2.5)
4. `xgboost_football_corners.pkl` (Target: Total Corners Over 9.5)

All artifacts are persisted to the `models/` directory.
"""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Reconfigure stdout for Windows console UTF-8 compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ml_pipeline import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("train_football")

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

FOOTBALL_FEATURE_NAMES = [
    "form_pts_diff",
    "goal_diff_diff",
    "home_atk_vs_away_def",
    "away_atk_vs_home_def",
    "prob_h",
    "prob_d",
    "prob_a",
    "implied_prob_1x",
    "odds_ratio_h_a",
    "bookmaker_margin",
]


# --------------------------------------------------------------------------- #
# Evaluation Metrics
# --------------------------------------------------------------------------- #
def calc_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    if len(y_t) == 0:
        return 0.0
    return float(np.mean(y_t == y_p))


def calc_log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """Compute binary cross-entropy (log loss)."""
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.clip(np.asarray(y_prob, dtype=np.float64), eps, 1.0 - eps)
    if len(y_t) == 0:
        return 0.0
    loss = -np.mean(y_t * np.log(y_p) + (1.0 - y_t) * np.log(1.0 - y_p))
    return float(loss)


def calc_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute exact ROC-AUC via Wilcoxon-Mann-Whitney rank-sum statistic."""
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


def evaluate_predictions(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Compute accuracy, log loss, and ROC-AUC for binary predictions."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": calc_accuracy(y_true, y_pred),
        "log_loss": calc_log_loss(y_true, y_prob),
        "roc_auc": calc_roc_auc(y_true, y_prob),
    }


def train_and_save_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    output_path: Path,
    model_label: str,
    n_estimators: int = 120,
    max_depth: int = 3,
    learning_rate: float = 0.05,
    random_state: int = 42,
) -> Dict[str, float]:
    """Fit XGBoost model, evaluate test set, persist to file, and return metrics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_predictions(y_test, y_prob)
    log.info(
        "%-32s | Accuracy: %.4f | LogLoss: %.4f | ROC-AUC: %.4f",
        model_label,
        metrics["accuracy"],
        metrics["log_loss"],
        metrics["roc_auc"],
    )
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    return metrics


# --------------------------------------------------------------------------- #
# Football Training Engine (4 Models)
# --------------------------------------------------------------------------- #
def train_football(
    csv_path: str | Path = DATA_DIR / "football.csv",
    split_ratio: float = 0.8,
) -> Dict[str, Dict[str, float]]:
    """Train 4 separate XGBoost models for Football:
    1. xgboost_football_1x.pkl      (Double Chance 1X)
    2. xgboost_football_hcp.pkl     (Handicap Cover against synthesized spread)
    3. xgboost_football_ou.pkl      (Total Goals Over/Under 2.5)
    4. xgboost_football_corners.pkl (Total Corners Over 9.5)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        fallback = DATA_DIR / "Football" / "ginf.csv"
        if fallback.exists():
            csv_path = fallback
        else:
            raise FileNotFoundError(f"Football CSV not found at {csv_path} or {fallback}")

    log.info("Loading Football dataset from %s", csv_path)
    df = pd.read_csv(csv_path)

    required = ["date", "ht", "at", "fthg", "ftag", "odd_h", "odd_d", "odd_a"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Football CSV missing columns: {missing}")

    df = df.dropna(subset=required).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Check for events.csv to extract actual corner kick counts
    events_path = DATA_DIR / "Football" / "events.csv"
    corner_dict: Dict[str, int] = {}
    if events_path.exists():
        log.info("Found %s. Extracting corner kick counts...", events_path)
        try:
            corners_df = pd.read_csv(events_path, usecols=["id_odsp", "event_type"])
            corners_df = corners_df[corners_df["event_type"] == 2]  # event_type 2 = Corner kick
            corner_counts = corners_df.groupby("id_odsp").size()
            corner_dict = corner_counts.to_dict()
            log.info("Extracted actual corner counts for %d matches from events.csv", len(corner_dict))
        except Exception as exc:
            log.warning("Could not read events.csv (%s). Using Poisson fallback.", exc)
    else:
        log.info("events.csv not found in data/Football/. Synthesizing baseline corner counts.")

    log.info("Extracting Football features for %d matches...", len(df))
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
        t_1x = 1 if fthg >= ftag else 0
        targets_1x.append(t_1x)

        # Target 2: Handicap (Home covers spread synthesized from goal diffs)
        spread = np.round((h_gd - a_gd) * 0.5 * 2.0) / 2.0
        t_hcp = 1 if (fthg - ftag) > spread else 0
        targets_hcp.append(t_hcp)

        # Target 3: Over/Under 2.5 Goals
        t_ou = 1 if (fthg + ftag) > 2.5 else 0
        targets_ou.append(t_ou)

        # Target 4: Corners Over 9.5
        match_id_odsp = str(row.get("id_odsp", ""))
        if match_id_odsp in corner_dict:
            tot_corners = corner_dict[match_id_odsp]
        else:
            # Poisson corner synthesizer based on attacking pace & goal volume
            lam_corners = max(4.0, 5.0 + 1.8 * (h_scored + a_scored) + 1.2 * max(0.0, h_gd))
            tot_corners = int(rng.poisson(lam_corners))
        t_corners = 1 if tot_corners > 9.5 else 0
        targets_corners.append(t_corners)

        # History tracking
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

    metrics_dict: Dict[str, Dict[str, float]] = {}

    log.info("--- Training Football Models (Train: %d, Test: %d) ---", len(X_train), len(X_test))

    # 1. Football 1X (Home Win or Draw)
    y_1x = np.array(targets_1x)
    m_1x = train_and_save_model(
        X_train, y_1x[:split_idx], X_test, y_1x[split_idx:],
        MODELS_DIR / "xgboost_football_1x.pkl", "Football 1X (Home/Draw)",
    )
    metrics_dict["football_1x"] = m_1x
    # Keep legacy names in sync for system-wide compatibility
    shutil.copyfile(MODELS_DIR / "xgboost_football_1x.pkl", MODELS_DIR / "xgboost_football.pkl")
    shutil.copyfile(MODELS_DIR / "xgboost_football_1x.pkl", MODELS_DIR / "xgboost_football_1x_v1.pkl")

    # 2. Football Handicap (Goal Spread Cover)
    y_hcp = np.array(targets_hcp)
    m_hcp = train_and_save_model(
        X_train, y_hcp[:split_idx], X_test, y_hcp[split_idx:],
        MODELS_DIR / "xgboost_football_hcp.pkl", "Football Handicap Cover",
    )
    metrics_dict["football_hcp"] = m_hcp

    # 3. Football Over/Under 2.5 Goals
    y_ou = np.array(targets_ou)
    m_ou = train_and_save_model(
        X_train, y_ou[:split_idx], X_test, y_ou[split_idx:],
        MODELS_DIR / "xgboost_football_ou.pkl", "Football Over/Under 2.5",
    )
    metrics_dict["football_ou"] = m_ou

    # 4. Football Corners Over 9.5
    y_c = np.array(targets_corners)
    m_c = train_and_save_model(
        X_train, y_c[:split_idx], X_test, y_c[split_idx:],
        MODELS_DIR / "xgboost_football_corners.pkl", "Football Corners Over 9.5",
    )
    metrics_dict["football_corners"] = m_c

    return metrics_dict


# --------------------------------------------------------------------------- #
# Main Execution Entrypoint
# --------------------------------------------------------------------------- #
def main():
    print("\n" + "=" * 88)
    print("PIKMATRIX FOOTBALL MODEL TRAINING ENGINE (TASK 1)")
    print("Training 4 High-Performance XGBoost Models ONLY")
    print("=" * 88)

    metrics = train_football()

    # Formatted Summary Table
    print("\n" + "=" * 88)
    print("TEST SET EVALUATION SUMMARY (4 FOOTBALL MODELS)")
    print("=" * 88)
    print(f"{'Model Artifact':<32} | {'Sport':<10} | {'Market Target':<24} | {'Accuracy':<9} | {'LogLoss':<8} | {'ROC-AUC':<7}")
    print("-" * 88)

    model_metadata = {
        "football_1x": ("Football", "1X (Home/Draw)"),
        "football_hcp": ("Football", "Goal Handicap Cover"),
        "football_ou": ("Football", "Total Goals > 2.5"),
        "football_corners": ("Football", "Total Corners > 9.5"),
    }

    for key, (sport, market) in model_metadata.items():
        if key not in metrics:
            continue
        m = metrics[key]
        artifact_name = f"xgboost_{key}.pkl"
        print(f"{artifact_name:<32} | {sport:<10} | {market:<24} | {m['accuracy']:<9.4f} | {m['log_loss']:<8.4f} | {m['roc_auc']:<7.4f}")

    print("=" * 88)

    # Verify created artifacts on disk
    print("\nVerifying artifacts in models/:")
    expected_files = [f"xgboost_{k}.pkl" for k in model_metadata.keys()] + [
        "xgboost_football.pkl",
        "xgboost_football_1x_v1.pkl",
    ]

    for name in expected_files:
        p = MODELS_DIR / name
        status = "OK" if p.exists() else "MISSING"
        size_kb = (p.stat().st_size / 1024) if p.exists() else 0
        print(f"  ✓ {name:<32}: [{status}] ({size_kb:.1f} KB)")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()
