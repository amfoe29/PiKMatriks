"""Multi-Sport Machine Learning Model Training Engine (Phase 6 Expansion).

Trains 10 sport-and-market-specific XGBoost models for:
1. Football (Soccer) - 10 features:
   - `xgboost_football_1x.pkl` (Target: Home Win or Draw)
   - `xgboost_football_hcp.pkl` (Target: Home Cover against spread)
   - `xgboost_football_ou.pkl` (Target: Total Goals Over 2.5)
   - `xgboost_football_corners.pkl` (Target: Total Corners Over 9.5)

2. Basketball (NBA) - 10 features:
   - `xgboost_basketball_ml.pkl` (Target: Home Win)
   - `xgboost_basketball_hcp.pkl` (Target: Home Point Spread Cover)
   - `xgboost_basketball_ou.pkl` (Target: Total Points Over/Under)

3. Tennis (ATP) - 11 features:
   - `xgboost_tennis_ml.pkl` (Target: Player 1 Win)
   - `xgboost_tennis_hcp.pkl` (Target: Player 1 covers Game Handicap)
   - `xgboost_tennis_ou.pkl` (Target: Match Total Games Over/Under)

Artifacts saved to `models/`.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import pickle
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Reconfigure stdout for Windows console UTF-8 compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.ml_pipeline import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_models")

PROJECT_ROOT = Path(__file__).resolve().parent
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

BASKETBALL_FEATURE_NAMES = [
    "home_win_rate_10",
    "away_win_rate_10",
    "win_rate_diff",
    "home_margin_10",
    "away_margin_10",
    "margin_diff",
    "home_off_vs_away_def",
    "away_off_vs_home_def",
    "scoring_power_diff",
    "season",
]

TENNIS_FEATURE_NAMES = [
    "rank_diff",
    "log_rank_ratio",
    "rank_pts_diff",
    "p1_recent_win_rate",
    "p2_recent_win_rate",
    "recent_win_rate_diff",
    "age_diff",
    "surface_hard",
    "surface_clay",
    "surface_grass",
    "best_of",
]


# --------------------------------------------------------------------------- #
# Evaluation Metrics (pure NumPy/Python)
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
        "%-30s | Accuracy: %.4f | LogLoss: %.4f | ROC-AUC: %.4f",
        model_label,
        metrics["accuracy"],
        metrics["log_loss"],
        metrics["roc_auc"],
    )
    with open(output_path, "wb") as f:
        pickle.dump(model, f)
    return metrics


# --------------------------------------------------------------------------- #
# 1. Football Multi-Market & Corners Training
# --------------------------------------------------------------------------- #
def train_football(
    csv_path: str | Path = DATA_DIR / "football.csv",
    split_ratio: float = 0.8,
) -> Dict[str, Dict[str, float]]:
    """Train 4 separate XGBoost models for Football:
    1. xgboost_football_1x.pkl (Double Chance 1X)
    2. xgboost_football_hcp.pkl (Handicap Cover against synthesized spread)
    3. xgboost_football_ou.pkl (Total Goals Over/Under 2.5)
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

    # Check for events.csv to extract corner kicks
    events_path = DATA_DIR / "Football" / "events.csv"
    corner_dict: Dict[str, int] = {}
    if events_path.exists():
        log.info("Found %s. Extracting corner kick counts...", events_path)
        try:
            corners_df = pd.read_csv(events_path, usecols=["id_odsp", "event_type"])
            corners_df = corners_df[corners_df["event_type"] == 2]
            corner_counts = corners_df.groupby("id_odsp").size()
            corner_dict = corner_counts.to_dict()
            log.info("Extracted corner counts for %d matches from events.csv", len(corner_dict))
        except Exception as exc:
            log.warning("Could not read events.csv (%s). Using Poisson fallback.", exc)
    else:
        log.info("events.csv not found in data/Football/. Engaging Poisson corner synthesizer.")

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
            # Poisson corner synthesizer: attacking pace & goal volume
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

    # 1. Football 1X
    y_1x = np.array(targets_1x)
    m_1x = train_and_save_model(
        X_train, y_1x[:split_idx], X_test, y_1x[split_idx:],
        MODELS_DIR / "xgboost_football_1x.pkl", "Football 1X (Home/Draw)",
    )
    metrics_dict["football_1x"] = m_1x
    # Keep legacy xgboost_football.pkl in sync
    shutil.copyfile(MODELS_DIR / "xgboost_football_1x.pkl", MODELS_DIR / "xgboost_football.pkl")

    # 2. Football Handicap
    y_hcp = np.array(targets_hcp)
    m_hcp = train_and_save_model(
        X_train, y_hcp[:split_idx], X_test, y_hcp[split_idx:],
        MODELS_DIR / "xgboost_football_hcp.pkl", "Football Handicap Cover",
    )
    metrics_dict["football_hcp"] = m_hcp

    # 3. Football Over/Under 2.5
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
# 2. Basketball Multi-Market Training
# --------------------------------------------------------------------------- #
def train_basketball(
    csv_path: str | Path = DATA_DIR / "basketball.csv",
    split_ratio: float = 0.8,
) -> Dict[str, Dict[str, float]]:
    """Train 3 separate XGBoost models for Basketball:
    1. xgboost_basketball_ml.pkl (Home Win Moneyline)
    2. xgboost_basketball_hcp.pkl (Home Point Spread Cover)
    3. xgboost_basketball_ou.pkl (Total Points Over/Under)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        fallback = DATA_DIR / "Basketball" / "games.csv"
        if fallback.exists():
            csv_path = fallback
        else:
            raise FileNotFoundError(f"Basketball CSV not found at {csv_path} or {fallback}")

    log.info("Loading Basketball dataset from %s", csv_path)
    df = pd.read_csv(csv_path)

    required = ["GAME_DATE_EST", "HOME_TEAM_ID", "VISITOR_TEAM_ID", "PTS_home", "PTS_away", "HOME_TEAM_WINS"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Basketball CSV missing columns: {missing}")

    df = df.dropna(subset=required).copy()
    df["GAME_DATE_EST"] = pd.to_datetime(df["GAME_DATE_EST"])
    df = df.sort_values("GAME_DATE_EST").reset_index(drop=True)

    log.info("Extracting Basketball features for %d games...", len(df))
    team_history: Dict[int, List[Tuple[int, float, float, float]]] = {}
    features = []
    targets_ml = []
    targets_hcp = []
    targets_ou = []

    for _, row in df.iterrows():
        hid = int(row["HOME_TEAM_ID"])
        aid = int(row["VISITOR_TEAM_ID"])

        h_last = team_history.get(hid, [])[-10:]
        a_last = team_history.get(aid, [])[-10:]

        h_wr = float(np.mean([x[0] for x in h_last])) if h_last else 0.5
        a_wr = float(np.mean([x[0] for x in a_last])) if a_last else 0.5

        h_margin = float(np.mean([x[1] for x in h_last])) if h_last else 0.0
        a_margin = float(np.mean([x[1] for x in a_last])) if a_last else 0.0

        h_scored = float(np.mean([x[2] for x in h_last])) if h_last else 103.0
        h_conceded = float(np.mean([x[3] for x in h_last])) if h_last else 101.0
        a_scored = float(np.mean([x[2] for x in a_last])) if a_last else 100.0
        a_conceded = float(np.mean([x[3] for x in a_last])) if a_last else 103.0

        season_val = float(row["SEASON"]) if "SEASON" in row and pd.notnull(row["SEASON"]) else 2020.0

        feat = [
            h_wr,
            a_wr,
            h_wr - a_wr,
            h_margin,
            a_margin,
            h_margin - a_margin,
            h_scored - a_conceded,
            a_scored - h_conceded,
            h_scored - a_scored,
            season_val,
        ]
        features.append(feat)

        pts_h = float(row["PTS_home"])
        pts_a = float(row["PTS_away"])
        target_ml = int(row["HOME_TEAM_WINS"])
        targets_ml.append(target_ml)

        # Target 2: Point Spread Cover
        spread = np.round((h_margin - a_margin) * 0.5 * 2.0) / 2.0
        t_hcp = 1 if (pts_h - pts_a) > spread else 0
        targets_hcp.append(t_hcp)

        # Target 3: Total Points Over/Under
        base_total = 0.5 * ((h_scored + a_conceded) + (a_scored + h_conceded))
        t_ou = 1 if (pts_h + pts_a) > base_total else 0
        targets_ou.append(t_ou)

        diff_h = pts_h - pts_a
        h_win = 1 if target_ml == 1 else 0
        team_history.setdefault(hid, []).append((h_win, diff_h, pts_h, pts_a))
        team_history.setdefault(aid, []).append((1 - h_win, -diff_h, pts_a, pts_h))

    X = pd.DataFrame(features, columns=BASKETBALL_FEATURE_NAMES)
    split_idx = int(split_ratio * len(X))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]

    metrics_dict: Dict[str, Dict[str, float]] = {}

    log.info("--- Training Basketball Models (Train: %d, Test: %d) ---", len(X_train), len(X_test))

    # 1. Basketball Moneyline
    y_ml = np.array(targets_ml)
    m_ml = train_and_save_model(
        X_train, y_ml[:split_idx], X_test, y_ml[split_idx:],
        MODELS_DIR / "xgboost_basketball_ml.pkl", "Basketball Moneyline",
        max_depth=4,
    )
    metrics_dict["basketball_ml"] = m_ml
    shutil.copyfile(MODELS_DIR / "xgboost_basketball_ml.pkl", MODELS_DIR / "xgboost_basketball.pkl")

    # 2. Basketball Handicap
    y_hcp = np.array(targets_hcp)
    m_hcp = train_and_save_model(
        X_train, y_hcp[:split_idx], X_test, y_hcp[split_idx:],
        MODELS_DIR / "xgboost_basketball_hcp.pkl", "Basketball Point Spread Cover",
        max_depth=3,
    )
    metrics_dict["basketball_hcp"] = m_hcp

    # 3. Basketball Over/Under
    y_ou = np.array(targets_ou)
    m_ou = train_and_save_model(
        X_train, y_ou[:split_idx], X_test, y_ou[split_idx:],
        MODELS_DIR / "xgboost_basketball_ou.pkl", "Basketball Total Points Over",
        max_depth=3,
    )
    metrics_dict["basketball_ou"] = m_ou

    return metrics_dict


# --------------------------------------------------------------------------- #
# 3. Tennis Multi-Market Training
# --------------------------------------------------------------------------- #
def parse_tennis_score(score_str: Any) -> Tuple[Optional[int], Optional[int]]:
    """Parse ATP score string (e.g. '6-4 6-3', '7-6(5) 6-4') to extract
    (total_games, winner_game_margin).
    """
    if not isinstance(score_str, str) or not score_str.strip():
        return None, None
    w_games, l_games = 0, 0
    set_patterns = re.findall(r"(\d+)-(\d+)", score_str)
    if not set_patterns:
        return None, None
    for w_str, l_str in set_patterns:
        w_games += int(w_str)
        l_games += int(l_str)
    return w_games + l_games, w_games - l_games


def train_tennis(
    csv_path: str | Path = DATA_DIR / "tennis.csv",
    split_ratio: float = 0.8,
    random_seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    """Train 3 separate XGBoost models for Tennis:
    1. xgboost_tennis_ml.pkl (Player 1 Win Moneyline)
    2. xgboost_tennis_hcp.pkl (Player 1 covers Game Handicap)
    3. xgboost_tennis_ou.pkl (Match Total Games Over/Under)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        fallback_files = sorted(glob.glob(str(DATA_DIR / "Tennis" / "atp_matches_*.csv")))
        if fallback_files:
            log.info("Concatenating fallback ATP match files: %s", fallback_files)
            df = pd.concat([pd.read_csv(f) for f in fallback_files], ignore_index=True)
        else:
            raise FileNotFoundError(f"Tennis CSV not found at {csv_path}")
    else:
        log.info("Loading Tennis dataset from %s", csv_path)
        df = pd.read_csv(csv_path)

    required = ["tourney_date", "winner_name", "loser_name", "winner_rank", "loser_rank", "surface"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Tennis CSV missing column: {c}")

    match_num_col = "match_num" if "match_num" in df.columns else "tourney_date"
    df = df.sort_values(["tourney_date", match_num_col]).reset_index(drop=True)

    log.info("Extracting Tennis features for %d matches...", len(df))
    rng = np.random.RandomState(random_seed)
    player_history: Dict[str, List[int]] = {}
    features = []
    targets_ml = []
    targets_hcp = []
    targets_ou = []

    for _, row in df.iterrows():
        w_name = str(row["winner_name"])
        l_name = str(row["loser_name"])

        w_rank = float(row["winner_rank"]) if pd.notnull(row["winner_rank"]) else 500.0
        l_rank = float(row["loser_rank"]) if pd.notnull(row["loser_rank"]) else 500.0

        w_pts = float(row["winner_rank_points"]) if ("winner_rank_points" in row and pd.notnull(row["winner_rank_points"])) else 0.0
        l_pts = float(row["loser_rank_points"]) if ("loser_rank_points" in row and pd.notnull(row["loser_rank_points"])) else 0.0

        w_age = float(row["winner_age"]) if ("winner_age" in row and pd.notnull(row["winner_age"])) else 26.0
        l_age = float(row["loser_age"]) if ("loser_age" in row and pd.notnull(row["loser_age"])) else 26.0

        # Parse match score
        total_g, g_margin = parse_tennis_score(row.get("score", ""))
        best_of = float(row["best_of"]) if ("best_of" in row and pd.notnull(row["best_of"])) else 3.0

        if total_g is None:
            total_g = 22 if best_of == 3.0 else 36
            g_margin = 4

        # Symmetrization: 50% chance Player 1 is winner, 50% Player 1 is loser
        coin = rng.rand() > 0.5
        if coin:
            p1_name, p2_name = w_name, l_name
            p1_rank, p2_rank = w_rank, l_rank
            p1_pts, p2_pts = w_pts, l_pts
            p1_age, p2_age = w_age, l_age
            target_ml = 1
            p1_margin = g_margin
        else:
            p1_name, p2_name = l_name, w_name
            p1_rank, p2_rank = l_rank, w_rank
            p1_pts, p2_pts = l_pts, w_pts
            p1_age, p2_age = l_age, w_age
            target_ml = 0
            p1_margin = -g_margin

        p1_last = player_history.get(p1_name, [])[-10:]
        p2_last = player_history.get(p2_name, [])[-10:]
        p1_wr = float(np.mean(p1_last)) if p1_last else 0.5
        p2_wr = float(np.mean(p2_last)) if p2_last else 0.5

        rank_diff = p2_rank - p1_rank
        log_ratio = float(np.log(max(1.0, p2_rank)) - np.log(max(1.0, p1_rank)))
        pts_diff = p1_pts - p2_pts
        age_diff = p1_age - p2_age

        surf = str(row["surface"]).lower() if pd.notnull(row["surface"]) else ""
        is_hard = 1.0 if "hard" in surf else 0.0
        is_clay = 1.0 if "clay" in surf else 0.0
        is_grass = 1.0 if "grass" in surf else 0.0

        feat = [
            rank_diff,
            log_ratio,
            pts_diff,
            p1_wr,
            p2_wr,
            p1_wr - p2_wr,
            age_diff,
            is_hard,
            is_clay,
            is_grass,
            best_of,
        ]
        features.append(feat)
        targets_ml.append(target_ml)

        # Target 2: Game Handicap Cover
        spread = np.round((log_ratio * 1.2) * 2.0) / 2.0
        t_hcp = 1 if p1_margin > spread else 0
        targets_hcp.append(t_hcp)

        # Target 3: Total Games Over/Under
        baseline_total = 36.5 if best_of == 5.0 else 22.5
        t_ou = 1 if total_g > baseline_total else 0
        targets_ou.append(t_ou)

        player_history.setdefault(w_name, []).append(1)
        player_history.setdefault(l_name, []).append(0)

    X = pd.DataFrame(features, columns=TENNIS_FEATURE_NAMES)
    split_idx = int(split_ratio * len(X))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]

    metrics_dict: Dict[str, Dict[str, float]] = {}

    log.info("--- Training Tennis Models (Train: %d, Test: %d) ---", len(X_train), len(X_test))

    # 1. Tennis Moneyline
    y_ml = np.array(targets_ml)
    m_ml = train_and_save_model(
        X_train, y_ml[:split_idx], X_test, y_ml[split_idx:],
        MODELS_DIR / "xgboost_tennis_ml.pkl", "Tennis Player 1 Win (ML)",
    )
    metrics_dict["tennis_ml"] = m_ml
    shutil.copyfile(MODELS_DIR / "xgboost_tennis_ml.pkl", MODELS_DIR / "xgboost_tennis.pkl")

    # 2. Tennis Handicap
    y_hcp = np.array(targets_hcp)
    m_hcp = train_and_save_model(
        X_train, y_hcp[:split_idx], X_test, y_hcp[split_idx:],
        MODELS_DIR / "xgboost_tennis_hcp.pkl", "Tennis Game Handicap Cover",
    )
    metrics_dict["tennis_hcp"] = m_hcp

    # 3. Tennis Over/Under Games
    y_ou = np.array(targets_ou)
    m_ou = train_and_save_model(
        X_train, y_ou[:split_idx], X_test, y_ou[split_idx:],
        MODELS_DIR / "xgboost_tennis_ou.pkl", "Tennis Total Games Over",
    )
    metrics_dict["tennis_ou"] = m_ou

    return metrics_dict


# --------------------------------------------------------------------------- #
# Dataset Inspection Helper
# --------------------------------------------------------------------------- #
def inspect_datasets(data_dir: Path = DATA_DIR) -> Dict[str, List[Dict[str, Any]]]:
    """Inspect all CSV files across data/ subfolders."""
    subfolders = ["Football", "Basketball", "Tennis"]
    inspection_results: Dict[str, List[Dict[str, Any]]] = {}

    print("=" * 80)
    print("DATASET INSPECTION REPORT")
    print("=" * 80)

    for sub in subfolders:
        folder_path = data_dir / sub
        inspection_results[sub] = []
        if not folder_path.exists():
            continue

        csv_files = sorted(glob.glob(str(folder_path / "*.csv")))
        print(f"\n--- Subfolder: data/{sub}/ ({len(csv_files)} CSV files) ---")

        for csv_path_str in csv_files:
            csv_path = Path(csv_path_str)
            try:
                sample_df = pd.read_csv(csv_path, nrows=2)
                row_count = sum(1 for _ in open(csv_path, "r", encoding="utf-8", errors="ignore")) - 1
                print(f"File: {csv_path.name:<25} | Shape: {row_count} rows x {len(sample_df.columns)} cols")
            except Exception as e:
                print(f"Error inspecting {csv_path.name}: {e}")

    print("=" * 80)
    return inspection_results


# --------------------------------------------------------------------------- #
# Main Multi-Market Trainer Runner
# --------------------------------------------------------------------------- #
def train_all_models() -> Dict[str, Dict[str, float]]:
    """Train all 10 multi-market models across Football, Basketball, and Tennis."""
    all_metrics: Dict[str, Dict[str, float]] = {}

    print("\n" + "=" * 85)
    print("PIKMATRIX MULTI-SPORT & MULTI-MARKET MODEL TRAINING ENGINE")
    print("Training 10 Specialized Machine Learning Models")
    print("=" * 85)

    # 1. Football (4 models)
    print("\n>>> [1/3] Training Football Models (1X, Handicap, Over/Under 2.5, Corners > 9.5)...")
    fb_metrics = train_football()
    all_metrics.update(fb_metrics)

    # 2. Basketball (3 models)
    print("\n>>> [2/3] Training Basketball Models (Moneyline, Point Spread, Over/Under)...")
    bb_metrics = train_basketball()
    all_metrics.update(bb_metrics)

    # 3. Tennis (3 models)
    print("\n>>> [3/3] Training Tennis Models (Moneyline, Game Handicap, Over/Under Games)...")
    tn_metrics = train_tennis()
    all_metrics.update(tn_metrics)

    # Summary Display
    print("\n" + "=" * 85)
    print("TEST SET EVALUATION SUMMARY (10 HIGH-VALUE MODELS)")
    print("=" * 85)
    print(f"{'Model Artifact':<32} | {'Sport':<10} | {'Market Target':<22} | {'Acc':<7} | {'LogLoss':<8} | {'AUC':<7}")
    print("-" * 85)

    model_metadata = {
        "football_1x": ("Football", "1X (Home/Draw)"),
        "football_hcp": ("Football", "Goal Handicap Cover"),
        "football_ou": ("Football", "Total Goals > 2.5"),
        "football_corners": ("Football", "Total Corners > 9.5"),
        "basketball_ml": ("Basketball", "Home Win (Moneyline)"),
        "basketball_hcp": ("Basketball", "Point Spread Cover"),
        "basketball_ou": ("Basketball", "Total Points Over"),
        "tennis_ml": ("Tennis", "Player 1 Win (ML)"),
        "tennis_hcp": ("Tennis", "Game Handicap Cover"),
        "tennis_ou": ("Tennis", "Total Games Over"),
    }

    for key, (sport, market) in model_metadata.items():
        m = all_metrics.get(key, {"accuracy": 0.0, "log_loss": 0.0, "roc_auc": 0.0})
        artifact_name = f"xgboost_{key}.pkl"
        print(f"{artifact_name:<32} | {sport:<10} | {market:<22} | {m['accuracy']:<7.4f} | {m['log_loss']:<8.4f} | {m['roc_auc']:<7.4f}")

    print("=" * 85)

    # Verify created artifacts
    print("\nVerifying artifacts in models/:")
    expected_files = [f"xgboost_{k}.pkl" for k in model_metadata.keys()] + [
        "xgboost_football.pkl",
        "xgboost_basketball.pkl",
        "xgboost_tennis.pkl",
    ]
    for name in expected_files:
        p = MODELS_DIR / name
        status = "OK" if p.exists() else "MISSING"
        size_kb = (p.stat().st_size / 1024) if p.exists() else 0
        print(f"  ✓ {name:<32}: [{status}] ({size_kb:.1f} KB)")
    print("=" * 85)

    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Multi-Sport & Multi-Market Model Training Engine")
    parser.add_argument("--inspect", action="store_true", help="Inspect datasets in data/ folder")
    parser.add_argument("--train", action="store_true", help="Train all 10 models")
    parser.add_argument("--football", action="store_true", help="Train Football models only")
    parser.add_argument("--basketball", action="store_true", help="Train Basketball models only")
    parser.add_argument("--tennis", action="store_true", help="Train Tennis models only")

    args = parser.parse_args()

    if not (args.inspect or args.train or args.football or args.basketball or args.tennis):
        train_all_models()
        return

    if args.inspect:
        inspect_datasets()

    if args.football:
        train_football()

    if args.basketball:
        train_basketball()

    if args.tennis:
        train_tennis()

    if args.train:
        train_all_models()


if __name__ == "__main__":
    main()
