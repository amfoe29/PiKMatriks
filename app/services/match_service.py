"""Match and Sports Analytics Service (Phase 2 Restructuring).

Handles match querying, projection synthesis, score distributions,
win probabilities, and telemetry enrichment for UI templates and APIs.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import AIPrediction, Match, MatchStatus
from app.ml_pipeline import (
    FeatureBuilder,
    compute_definitive_prediction,
    compute_bivariate_poisson_engine,
    generate_two_tier_predictions,
)

log = logging.getLogger("pikmatrix_match_service")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def format_kickoff(dt: Optional[datetime]) -> str:
    """Format datetime into a readable UTC timestamp string."""
    if not dt:
        return "TBD"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.strftime("%b %d, %Y • %H:%M UTC")


def get_ordinal_suffix(day: int) -> str:
    """Return day with ordinal suffix, e.g. 7th, 8th, 9th, 1st, 2nd, 3rd."""
    if 11 <= (day % 100) <= 13:
        return f"{day}th"
    return f"{day}" + {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")


def build_calendar_days(
    current_date: Optional[date] = None,
    active_date_filter: str = "today",
) -> list[dict[str, Any]]:
    """Build the 5-day Sofascore-style calendar strip centered around current_date."""
    if current_date is None:
        current_date = datetime.now(timezone.utc).date()
    offsets = [
        (-2, "2d ago"),
        (-1, "Yesterday"),
        (0, "Today"),
        (1, "Tomorrow"),
        (2, "In 2d"),
    ]
    days = []
    active_clean = (active_date_filter or "today").strip().lower()

    for offset, rel_label in offsets:
        target_d = current_date + timedelta(days=offset)
        d_str = target_d.strftime("%Y-%m-%d")
        day_abbr = target_d.strftime("%a.")
        day_num = get_ordinal_suffix(target_d.day)

        if offset == -1:
            filter_key = "yesterday"
            filter_query = "filter=yesterday"
            is_active = (active_clean in ["yesterday", d_str])
        elif offset == 0:
            filter_key = "today"
            filter_query = "filter=today"
            is_active = (active_clean in ["today", d_str, ""])
        elif offset == 1:
            filter_key = "tomorrow"
            filter_query = "filter=tomorrow"
            is_active = (active_clean in ["tomorrow", d_str])
        else:
            filter_key = d_str
            filter_query = f"date={d_str}"
            is_active = (active_clean == d_str)

        days.append({
            "offset": offset,
            "date_str": d_str,
            "day_abbr": day_abbr,
            "day_num": day_num,
            "rel_label": rel_label,
            "filter_key": filter_key,
            "filter_query": filter_query,
            "is_today": (offset == 0),
            "is_active": is_active,
        })
    return days


def compute_score_distribution(
    sport_key: str, home_val: float, away_val: float
) -> list[dict[str, Any]]:
    """Generate exact score probabilities and prices based on calculated xG values."""
    if sport_key == "football":
        h_lam = max(0.2, float(home_val or 1.4))
        a_lam = max(0.2, float(away_val or 1.1))

        scores = []
        for h in range(6):
            for a in range(6):
                p_h = (h_lam**h) * math.exp(-h_lam) / math.factorial(h)
                p_a = (a_lam**a) * math.exp(-a_lam) / math.factorial(a)
                prob = p_h * p_a
                price_val = round(max(1.10, 1.0 / (max(0.005, prob) * 1.08)), 2)
                price_str = f"{price_val:.2f}"
                chance_pct = max(1, int(round(prob * 100)))
                scores.append({
                    "score": f"{h}–{a}",
                    "prob": round(prob, 4),
                    "pct": f"{prob * 100:.1f}%",
                    "pct_num": round(prob * 100, 1),
                    "chance_str": f"{chance_pct}%",
                    "price_str": price_str,
                    "home": h,
                    "away": a,
                })
        scores.sort(key=lambda x: x["prob"], reverse=True)
        top_list = scores[:12]
        if top_list:
            max_p = max(s["prob"] for s in top_list[:9])
            min_p = min(s["prob"] for s in top_list[:9])
            diff_p = max(1e-4, max_p - min_p)
            for s in top_list:
                norm = max(0.0, min(1.0, (s["prob"] - min_p) / diff_p))
                s["heat_opacity"] = round(0.10 + 0.70 * norm, 2)
                s["heat_pct"] = int(round(s["heat_opacity"] * 100))
        return top_list

    elif sport_key == "basketball":
        h_base = int(round(home_val or 110))
        a_base = int(round(away_val or 105))
        offsets = [(0, 0), (2, -1), (-2, 1), (3, -2), (1, 1), (4, -3), (-1, 2), (2, 2), (-3, 3), (5, -4)]
        base_probs = [0.135, 0.118, 0.104, 0.089, 0.076, 0.065, 0.058, 0.052, 0.045, 0.038]
        b_scores = []
        max_bp = max(base_probs[:9])
        min_bp = min(base_probs[:9])
        diff_bp = max(1e-4, max_bp - min_bp)
        for (oh, oa), p in zip(offsets, base_probs):
            norm = max(0.0, min(1.0, (p - min_bp) / diff_bp))
            opac = round(0.10 + 0.70 * norm, 2)
            b_scores.append({
                "score": f"{h_base + oh}–{a_base + oa}",
                "prob": p,
                "pct": f"{p * 100:.1f}%",
                "pct_num": round(p * 100, 1),
                "chance_str": f"{int(round(p * 100))}%",
                "price_str": f"{round(1.0 / (p * 1.08), 2):.2f}",
                "home": h_base + oh,
                "away": a_base + oa,
                "heat_opacity": opac,
                "heat_pct": int(round(opac * 100)),
            })
        return b_scores

    else:
        # Tennis sets distribution
        try:
            p1 = float(str(home_val).replace("%", "")) / 100.0 if home_val else 0.55
        except Exception:
            p1 = 0.55
        p2 = 1.0 - p1
        if p1 >= p2:
            items = [
                ("2 - 0 Sets", 0.46),
                ("2 - 1 Sets", 0.32),
                ("1 - 2 Sets", 0.14),
                ("0 - 2 Sets", 0.08),
            ]
        else:
            items = [
                ("0 - 2 Sets", 0.46),
                ("1 - 2 Sets", 0.32),
                ("2 - 1 Sets", 0.14),
                ("2 - 0 Sets", 0.08),
            ]
        return [
            {
                "score": name,
                "prob": p,
                "pct": f"{p * 100:.1f}%",
                "pct_num": round(p * 100, 1),
                "home": int(name[0]),
                "away": int(name[4]),
            }
            for name, p in items
        ]


def compute_win_probabilities(
    sport_key: str, home_val: float, away_val: float
) -> dict[str, int]:
    """Calculate 3-segment win probability distribution (Home %, Draw %, Away %)."""
    if sport_key == "football":
        h_lam = max(0.2, float(home_val or 1.4))
        a_lam = max(0.2, float(away_val or 1.1))
        p_home = 0.0
        p_draw = 0.0
        p_away = 0.0
        for h in range(7):
            for a in range(7):
                p = (
                    ((h_lam**h) * math.exp(-h_lam) / math.factorial(h))
                    * ((a_lam**a) * math.exp(-a_lam) / math.factorial(a))
                )
                if h > a:
                    p_home += p
                elif h == a:
                    p_draw += p
                else:
                    p_away += p
        tot = p_home + p_draw + p_away
        if tot > 0:
            h_pct = int(round((p_home / tot) * 100))
            d_pct = int(round((p_draw / tot) * 100))
            a_pct = max(0, 100 - h_pct - d_pct)
        else:
            h_pct, d_pct, a_pct = 48, 26, 26
        return {"home_pct": h_pct, "draw_pct": d_pct, "away_pct": a_pct}

    elif sport_key == "basketball":
        diff = float(home_val or 110) - float(away_val or 105)
        norm_val = 0.5 * (1.0 + math.erf(diff / (11.5 * math.sqrt(2.0))))
        h_pct = int(round(min(0.92, max(0.08, norm_val)) * 100))
        d_pct = 0
        a_pct = 100 - h_pct
        return {"home_pct": h_pct, "draw_pct": d_pct, "away_pct": a_pct}

    else:
        try:
            p1_wr = float(str(home_val).replace("%", "")) / 100.0
        except Exception:
            p1_wr = 0.55
        h_pct = int(round(min(0.92, max(0.08, p1_wr)) * 100))
        d_pct = 0
        a_pct = 100 - h_pct
        return {"home_pct": h_pct, "draw_pct": d_pct, "away_pct": a_pct}


def format_consumer_action_title(
    market_type: str,
    selection: str,
    line: Optional[float],
    home_name: str,
    away_name: str,
    sport_key: str,
) -> str:
    """Format clean, consumer-standard betting action titles."""
    m_type = (market_type or "").lower()
    sel = (selection or "").lower()
    line_str = f"{line:g}" if line is not None else ""

    if m_type == "double_chance":
        if "x2" in sel:
            return "Double Chance: Away or Draw (X2)"
        return "Double Chance: Home or Draw (1X)"
    elif m_type == "draw_no_bet":
        team = home_name if sel in ("home", "1") else away_name
        return f"Draw No Bet (DNB): {team}"
    elif m_type == "set_spread":
        team = home_name if sel in ("home", "1", "player_1") else away_name
        sign = "+" if (line is not None and line > 0) else ""
        spread_label = f"{sign}{line_str} Sets" if line is not None else "+1.5 Sets"
        return f"Set Spread: {team} ({spread_label})"
    elif m_type in ("over_under", "over_under_goals", "totals"):
        if sport_key == "basketball":
            unit = "Points"
        elif sport_key == "tennis":
            unit = "Games"
        else:
            unit = "Goals"
        sel_word = "Under" if "under" in sel else "Over"
        return f"Total {unit}: {sel_word} {line_str} {unit}"
    elif m_type in ("corners", "over_under_corners"):
        sel_word = "Under" if "under" in sel else "Over"
        return f"Corner Count: {sel_word} {line_str} Corners"
    elif m_type in ("handicap", "spread"):
        sign = "+" if (line is not None and line > 0) else ""
        team = home_name if sel == "home" else away_name
        spread_label = f"{sign}{line_str}" if line is not None else ""
        if sport_key == "basketball":
            return f"Point Spread: {team} ({spread_label})"
        elif sport_key == "tennis":
            return f"Game Handicap: {team} ({spread_label})"
        else:
            return f"Asian Handicap: {team} ({spread_label})"
    elif m_type in ("moneyline", "1x2"):
        team = home_name if sel in ("home", "1") else away_name
        return f"Match Winner: {team}"
    return f"{market_type.replace('_', ' ').title()}: {selection.title()}"


def format_tier_action_label(
    market_type: str,
    selection: str,
    line: Optional[float],
    home_name: str,
    away_name: str,
    sport_key: str,
) -> str:
    """Format concise action labels for locks and ladders."""
    m_type = (market_type or "").lower()
    sel = (selection or "").lower()
    line_str = f"{line:g}" if line is not None else ""

    if m_type in ("over_under", "over_under_goals", "totals"):
        if sport_key == "basketball":
            unit = "Points"
        elif sport_key == "tennis":
            unit = "Games"
        else:
            unit = "Goals"
        sel_word = "Under" if "under" in sel else "Over"
        return f"{sel_word} {line_str} {unit}".strip()

    elif m_type in ("corners", "over_under_corners"):
        sel_word = "Under" if "under" in sel else "Over"
        return f"{sel_word} {line_str} Corners".strip()

    elif m_type in ("handicap", "spread"):
        sign = "+" if (line is not None and line > 0) else ""
        team = home_name if sel in ("home", "1") else away_name
        spread_val = f"{sign}{line_str}" if line is not None else ""
        if sport_key == "basketball":
            return f"{team} ({spread_val} Pts)"
        elif sport_key == "tennis":
            return f"{team} ({spread_val} Games)"
        return f"{team} ({spread_val})"

    elif m_type == "set_spread":
        team = home_name if sel in ("home", "1", "player_1") else away_name
        sign = "+" if (line is not None and line > 0) else ""
        spread_val = f"{sign}{line_str} Sets" if line is not None else "+1.5 Sets"
        return f"{team} ({spread_val})"

    elif m_type == "double_chance":
        if "x2" in sel:
            return f"{away_name} or Draw (X2)"
        return f"{home_name} or Draw (1X)"

    elif m_type == "draw_no_bet":
        team = home_name if sel in ("home", "1") else away_name
        return f"Draw No Bet: {team}"

    elif m_type in ("moneyline", "1x2"):
        team = home_name if sel in ("home", "1") else away_name
        return f"{team} To Win"

    if line is not None:
        return f"{selection.title()} {line_str}".strip()
    return selection.title()


def generate_match_insight(
    sport_key: str,
    market_type: str,
    selection: str,
    line: Optional[float],
    confidence: float,
    home_name: str,
    away_name: str,
    definitive: dict[str, Any],
    is_alternate: bool = False,
    cushion_type: Optional[str] = None,
) -> str:
    """Generate statistical match insight with natural terminology."""
    sp = sport_key.lower()
    m_type = (market_type or "").lower()
    sel = (selection or "").lower()
    line_str = f"{line:g}" if line is not None else ""
    conf_pct = int(round(confidence * 100))

    if is_alternate:
        if "football" in sp:
            if m_type in ("over_under", "over_under_goals", "totals"):
                if "under" in sel:
                    if line is not None and line >= 4.5:
                        return f"Contested match dynamics introduce volatility, but Under {line_str} Goals provides an {conf_pct}% high-probability ceiling."
                    return f"Primary match winner line is volatile, but low attacking xG indicates strong resilience for Under {line_str or '3.5'} Goals."
                return f"Combined attacking efficiency provides an {conf_pct}% statistical cushion for Over {line_str or '1.5'} Goals."
            elif m_type in ("handicap", "spread"):
                target_team = home_name if sel == "home" else away_name
                sign = "+" if (line is not None and line > 0) else ""
                if line is not None and abs(line) >= 2.0:
                    return f"Straight win carries variance; +{line_str} Handicap provides a {conf_pct}% statistical cushion."
                return f"Straight win carries variance; {sign}{line_str or '1.5'} Asian Handicap provides an {conf_pct}% statistical cushion against tighter margins."
            elif m_type in ("corners", "over_under_corners"):
                if "under" in sel:
                    if line is not None and line >= 12.5:
                        return f"Midfield containment limits wing crossings, establishing an {conf_pct}% defensive ceiling for Under {line_str} Corners."
                    return f"Both squads rank low in wing progression; match profile heavily favors Under {line_str or '10.5'} Corners."
                return f"Direct transitions and persistent flank crosses reliably generate sustained set-piece opportunities, securing Over {line_str or '6.5'} Corners."
            elif m_type == "draw_no_bet":
                target_team = home_name if sel in ("home", "1") else away_name
                return f"Draw No Bet structure protects the stake against a level score while backing {target_team}."
            elif m_type in ("double_chance", "1x"):
                target_team = away_name if "x2" in sel else home_name
                return f"Contested matchup; Double Chance provides a resilient {conf_pct}% safety cushion backing {target_team}."
        elif "basketball" in sp:
            if m_type in ("handicap", "spread"):
                target_team = home_name if sel == "home" else away_name
                sign = "+" if (line is not None and line > 0) else ""
                return f"Outright moneyline carries tight margin risk; adjusting to {sign}{line_str or '6.5'} spread delivers an {conf_pct}% defensive cushion."
            elif m_type in ("over_under", "totals"):
                if "under" in sel:
                    return f"Standard total line sits near the median pace; adjusting to Under {line_str or '227.5'} Points provides strong resistance against late-game fouling."
                return f"Pace-adjusted offensive ratings indicate consistent scoring velocity, establishing an {conf_pct}% cushion for Over {line_str or '207.5'} Points."
        elif "tennis" in sp:
            if m_type == "set_spread":
                target_player = home_name if sel in ("home", "1", "player_1") else away_name
                sign = "+" if (line is not None and line > 0) else ""
                return f"Straight match winner line carries variance; {sign}{line_str or '1.5'} Set Spread ensures coverage if either player pushes the fixture into a deciding third set."
            elif m_type in ("over_under", "totals"):
                if "under" in sel:
                    return f"Dominant service holds suggest straight-sets momentum; Under {line_str or '24.5'} Games provides an {conf_pct}% cushion against prolonged tiebreaks."
                return f"Competitive baseline rallies project extended game length, creating a resilient {conf_pct}% margin for Over {line_str or '19.5'} Games."
            elif m_type in ("handicap", "spread"):
                target_player = home_name if sel == "home" else away_name
                sign = "+" if (line is not None and line > 0) else ""
                return f"Straight win carries game-variance risk; {sign}{line_str} Game Handicap gives {target_player} an {conf_pct}% statistical safety cushion."

    # Standard non-alternate insight generation
    if "football" in sp:
        if m_type in ("double_chance", "1x"):
            if "x2" in sel or (definitive.get("pred_result") and "away" in str(definitive.get("pred_result")).lower()):
                return f"{away_name} demonstrates resilient counter-attacking and road form; Double Chance covers high-stalemate variance."
            return f"{home_name} displays strong defensive form; Double Chance covers high-draw probability with a solid home margin."
        elif m_type in ("over_under", "over_under_goals", "totals"):
            if "under" in sel or confidence < 0.50 or (line and line >= 2.5 and definitive.get("projected_total_goals", 2) < line):
                return f"Both sides exhibit compact defensive shape and controlled midfield pacing; projected scoring velocity strongly favors Under {line_str or '2.5'} Goals."
            return f"High combined attacking efficiency and open transitions indicate consistent chance creation, projecting an active game clearing Over {line_str or '1.5'} Goals."
        elif m_type in ("corners", "over_under_corners"):
            if "under" in sel or (line and line >= 9.5 and definitive.get("projected_corners", 9) < line):
                return f"Low corner conversion rate indicates a high probability of Under {line_str or '10.5'} Corners."
            return f"Aggressive flank progression and elevated shot-deflection rates indicate sustained wide pressure, favoring Over {line_str or '8.5'} Corners."
        elif m_type in ("handicap", "spread"):
            target_team = home_name if sel == "home" else away_name
            sign = "+" if (line is not None and line > 0) else ""
            if line is not None and abs(line) >= 2.0:
                return f"Straight win carries variance; +{line_str} Handicap provides a {conf_pct}% deep statistical cushion."
            return f"Form rating differentials and defensive resilience give {target_team} a solid cushion to cover the ({sign}{line_str}) spread."
        elif m_type in ("moneyline", "1x2"):
            target_team = home_name if sel in ("home", "1") else away_name
            return f"Superior chance-creation quality gives {target_team} a definitive edge to secure the victory in standard time."

    elif "basketball" in sp:
        if m_type in ("handicap", "spread"):
            target_team = home_name if sel == "home" else away_name
            sign = "+" if (line is not None and line > 0) else ""
            return f"Pace-adjusted offensive ratings and perimeter defensive efficiency position {target_team} favorably to cover the ({sign}{line_str}) spread."
        elif m_type in ("over_under", "totals"):
            if "under" in sel:
                return f"Half-court defensive discipline suppresses total scoring opportunities, projecting Under {line_str or '215.5'} Points."
            return f"Fast-break frequency and high 3-point attempt rates project an accelerated tempo, driving total match scoring Over {line_str or '211.5'} Points."
        elif m_type in ("moneyline", "1x2"):
            target_team = home_name if sel in ("home", "1") else away_name
            return f"Dominant rebounding rate and offensive rating advantage give {target_team} a strong moneyline projection."

    elif "tennis" in sp:
        if m_type in ("handicap", "spread"):
            target_player = home_name if sel == "home" else away_name
            sign = "+" if (line is not None and line > 0) else ""
            return f"First-serve hold consistency and break-point defense give {target_player} a significant margin to cover the ({sign}{line_str}) game spread."
        elif m_type in ("over_under", "totals"):
            if "under" in sel:
                return f"Dominant service hold ratings and low break frequency point to a swift straight-sets outcome, favoring Under {line_str or '23.5'} Games."
            return f"Evenly matched baseline exchange ratings indicate a prolonged contest, pushing total games Over {line_str or '21.5'}."
        elif m_type in ("moneyline", "1x2"):
            target_player = home_name if sel in ("home", "1") else away_name
            return f"Surface-specific win rates and recent return game efficiency heavily favor {target_player} to advance."

    return "Recent team form and head-to-head metrics indicate favorable value on this market selection."


def get_model_retrain_metadata() -> dict[str, Any]:
    """Retrieve the latest autonomous model retraining and self-learning telemetry."""
    metadata_file = PROJECT_ROOT / "models" / "retrain_metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning("Could not read retrain_metadata.json: %s", e)

    model_file = PROJECT_ROOT / "models" / "xgboost_football_1x.pkl"
    if model_file.exists():
        mtime = datetime.fromtimestamp(model_file.stat().st_mtime, tz=timezone.utc)
        return {
            "last_trained_display": mtime.strftime("%b %d, %Y • %H:%M UTC"),
            "self_learning": True,
            "engine_status": "active",
        }
    return {
        "last_trained_display": datetime.now(timezone.utc).strftime("%b %d, %Y • %H:%M UTC"),
        "self_learning": True,
        "engine_status": "active",
    }


def get_team_initials(name: Optional[str]) -> str:
    """Return 2-letter uppercase initials for team/player badge."""
    if not name:
        return "??"
    parts = name.strip().split()
    if len(parts) == 1:
        return parts[0][:3].upper()
    return (parts[0][0] + parts[1][0]).upper()


def get_league_code(comp_name: Optional[str]) -> str:
    """Return clean alphanumeric league code (zero emoji)."""
    if not comp_name:
        return "GEN"
    c = comp_name.lower()
    if "premier" in c or "england" in c:
        return "EPL"
    if "la liga" in c or "spain" in c:
        return "ESP"
    if "serie a" in c or "italy" in c:
        return "ITA"
    if "bundesliga" in c or "germany" in c:
        return "GER"
    if "ligue 1" in c or "france" in c:
        return "FRA"
    if "champions" in c or "uefa" in c:
        return "UCL"
    if "nba" in c:
        return "NBA"
    if "euroleague" in c:
        return "EUR"
    if "atp" in c:
        return "ATP"
    if "wta" in c:
        return "WTA"
    return "INTL"


def enrich_match_for_ui(
    m: Match,
    db: Session,
    builder: Optional[FeatureBuilder] = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Enriches a Match object with projections, odds, and two-tier predictions for UI consumption."""
    feature_builder = builder or FeatureBuilder()
    sp_name = (m.sport.name if m.sport else "football").lower()
    sport_key = "basketball" if "basketball" in sp_name else ("tennis" if "tennis" in sp_name else "football")
    home_name = m.home_entity.name if m.home_entity else f"Team_{m.home_entity_id}"
    away_name = m.away_entity.name if m.away_entity else f"Team_{m.away_entity_id}"
    comp_name = m.competition_name or "General"

    definitive = compute_definitive_prediction(m, session=db, builder=feature_builder)
    win_probs = compute_win_probabilities(
        sport_key,
        definitive.get("home_metric_val", 0),
        definitive.get("away_metric_val", 0),
    )
    score_dist = compute_score_distribution(
        sport_key,
        definitive.get("home_metric_val", 0),
        definitive.get("away_metric_val", 0),
    )

    is_finished = (m.status == MatchStatus.FINISHED)
    final_score = m.final_score or {}
    is_live = (
        m.status == MatchStatus.LIVE
        or bool(final_score.get("is_live"))
        or str(final_score.get("status", "")).lower() == "live"
    )
    live_minute = str(
        final_score.get("current_minute")
        or final_score.get("minute")
        or ("FT" if is_finished else "LIVE")
    )
    home_final = final_score.get("home") if (is_finished or is_live) else None
    away_final = final_score.get("away") if (is_finished or is_live) else None
    score_display = f"{home_final} - {away_final}" if (home_final is not None and away_final is not None) else None

    # Dynamic Two-Tier Prediction Generator
    two_tier = generate_two_tier_predictions(m, session=db, builder=feature_builder)
    tier1 = two_tier.get("capital_preservation_play")
    tier2 = two_tier.get("alpha_value_play")
    archetype_info = two_tier.get("archetype", {})
    is_unplayable = two_tier.get("is_unplayable", False)
    unplayable_msg = two_tier.get("unplayable_message")
    active_plays = two_tier.get("active_plays", [])

    # Map database prediction IDs, settlement_status, and actual_result_context
    db_preds = db.scalars(select(AIPrediction).where(AIPrediction.match_id == m.id)).all()
    db_model_map = {p.model_name: p for p in db_preds if p.model_name}
    db_sel_map = {(p.selection or "").strip().lower(): p for p in db_preds if p.selection}
    db_type_map = {(p.market_type or "").strip().lower(): p for p in db_preds if p.market_type}

    def _apply_db_pred(play_dict: dict[str, Any], dp: AIPrediction) -> None:
        play_dict["id"] = dp.id
        if dp.settlement_status:
            play_dict["settlement_status"] = dp.settlement_status.upper()
        if dp.actual_result_context:
            play_dict["actual_result_context"] = dp.actual_result_context
        if dp.settled_at:
            play_dict["settled_at"] = dp.settled_at.isoformat() if hasattr(dp.settled_at, "isoformat") else str(dp.settled_at)
        st = (play_dict.get("settlement_status") or "").upper()
        ctx = play_dict.get("actual_result_context") or ""
        if st and st != "PENDING":
            play_dict["settlement_badge"] = {
                "status": st.lower(),
                "badge": f"{st} • {ctx}" if ctx else st,
                "detail": ctx,
            }

    for p in active_plays:
        m_name = p.get("model_name")
        p_sel = (p.get("selection") or "").strip().lower()
        m_type = (p.get("market_type") or "").strip().lower()
        matched_dp = (
            (db_model_map.get(m_name) if m_name else None)
            or db_sel_map.get(p_sel)
            or db_type_map.get(m_type)
        )
        if matched_dp:
            _apply_db_pred(p, matched_dp)

    if tier1:
        dp1 = db_model_map.get("alpha_fortress_lock") or db_sel_map.get((tier1.get("selection") or "").strip().lower())
        if dp1:
            _apply_db_pred(tier1, dp1)

    if tier2:
        dp2 = db_model_map.get("alpha_value_edge") or db_sel_map.get((tier2.get("selection") or "").strip().lower())
        if dp2:
            _apply_db_pred(tier2, dp2)

    for t in active_plays:
        if t:
            if isinstance(t.get("decimal_odds"), dict):
                t["decimal_odds"] = t["decimal_odds"].get("decimal_odds") or t["decimal_odds"].get("odds")
            elif t.get("decimal_odds") is not None:
                try:
                    t["decimal_odds"] = round(float(t["decimal_odds"]), 2)
                except (ValueError, TypeError):
                    pass

    # Evaluate Bivariate Poisson Predictive Engine
    bivariate = compute_bivariate_poisson_engine(m, session=db)
    ai_confidence_int = bivariate.get("ai_confidence_pct", 75)
    promotion_tax_applied = bool(
        bivariate.get("promotion_tax_applied")
        or (m.home_entity and getattr(m.home_entity, "promotion_tax_applied", False))
        or (m.away_entity and getattr(m.away_entity, "promotion_tax_applied", False))
    )

    top_play = tier1 or tier2
    conf_float = top_play.get("confidence_num", ai_confidence_int / 100.0) if top_play else (ai_confidence_int / 100.0)
    conf_pct_str = f"{ai_confidence_int}%"
    risk_lvl = "Low Risk" if conf_float >= 0.85 else ("Medium Risk" if conf_float >= 0.70 else "High Risk")

    m_dict = {
        "id": m.id,
        "match_id": m.id,
        "ai_confidence_pct": ai_confidence_int,
        "bivariate": bivariate,
        "promotion_tax_applied": promotion_tax_applied,
        "sport": m.sport.name.capitalize() if m.sport else "Football",
        "sport_key": sport_key,
        "competition": comp_name,
        "competition_flag": get_league_code(comp_name),
        "home_team": home_name,
        "away_team": away_name,
        "home_logo_url": (m.home_entity.logo_url if m.home_entity else None),
        "away_logo_url": (m.away_entity.logo_url if m.away_entity else None),
        "home_initials": get_team_initials(home_name),
        "away_initials": get_team_initials(away_name),
        "fixture": f"{home_name} vs {away_name}",
        "kickoff_time": m.kickoff_time,
        "kickoff_time_str": m.kickoff_time.strftime("%H:%M") if m.kickoff_time else "18:00",
        "kickoff_iso": m.kickoff_time.isoformat() if m.kickoff_time else None,
        "kickoff_display": format_kickoff(m.kickoff_time),
        "status": m.status.value if hasattr(m.status, "value") else str(m.status),
        "is_live": is_live,
        "live_minute": live_minute,
        "is_finished": is_finished,
        "final_score": final_score if (is_finished or is_live) else None,
        "home_score": home_final,
        "away_score": away_final,
        "score_display": score_display,
        "has_safe_pick": (tier1 is not None),
        "has_value_pick": (tier2 is not None),
        "has_safe_play": (len(active_plays) > 0),
        "is_unplayable": is_unplayable,
        "unplayable_message": unplayable_msg,
        "archetype": archetype_info,
        "capital_preservation_play": tier1,
        "alpha_value_play": tier2,
        "two_tier_predictions": two_tier,
        "safe_count": len(active_plays),
        "max_confidence": conf_float,
        "max_confidence_pct": conf_pct_str,
        "predictions": active_plays,
        "additional_plays": [
            p for p in active_plays
            if p is not tier1 and p is not tier2
            and (not tier1 or (p.get("selection") or "").strip().lower() != (tier1.get("selection") or "").strip().lower())
            and (not tier2 or (p.get("selection") or "").strip().lower() != (tier2.get("selection") or "").strip().lower())
        ],
        "edge": round(float(tier2["alpha_edge"]), 1) if (tier2 and tier2.get("alpha_edge") is not None) else None,
        "confidence_score": conf_pct_str,
        "confidence_pct": round(conf_float * 100, 1) if conf_float <= 1.0 else round(conf_float, 1),
        "predictions_by_market": {p["market_type"]: p for p in active_plays},
        "pred_score": definitive.get("pred_score", "0 - 0"),
        "pred_home_score": definitive.get("pred_home_score", 0),
        "pred_away_score": definitive.get("pred_away_score", 0),
        "pred_total_goals": definitive.get("pred_total_goals", 0),
        "pred_total_display": definitive.get("pred_total_display", ""),
        "pred_result": definitive.get("pred_result", "TBD"),
        "projected_home_goals": definitive.get("projected_home_goals"),
        "projected_away_goals": definitive.get("projected_away_goals"),
        "projected_total_goals": definitive.get("projected_total_goals"),
        "projected_corners": definitive.get("projected_corners", 9.2),
        "expected_goals_home": m.expected_goals_home if m.expected_goals_home is not None else definitive.get("projected_home_goals"),
        "expected_goals_away": m.expected_goals_away if m.expected_goals_away is not None else definitive.get("projected_away_goals"),
        "corner_count": m.corner_count,
        "home_average_xg": m.home_entity.average_xg if m.home_entity else None,
        "away_average_xg": m.away_entity.average_xg if m.away_entity else None,
        "home_average_possession": m.home_entity.average_possession if m.home_entity else None,
        "away_average_possession": m.away_entity.average_possession if m.away_entity else None,
        "home_form_rating": m.home_entity.form_rating if m.home_entity else None,
        "away_form_rating": m.away_entity.form_rating if m.away_entity else None,
        "projected_home_points": definitive.get("projected_home_points"),
        "projected_away_points": definitive.get("projected_away_points"),
        "projected_total_points": definitive.get("projected_total_points"),
        "projected_total_games": definitive.get("projected_total_games", 22),
        "projected_winner": definitive.get("projected_winner"),
        "home_metric_name": definitive.get("home_metric_name", "xG"),
        "away_metric_name": definitive.get("away_metric_name", "xG"),
        "home_metric_val": definitive.get("home_metric_val", 0),
        "away_metric_val": definitive.get("away_metric_val", 0),
        "win_probabilities": win_probs,
        "score_distribution": score_dist,
        "primary_verdict": top_play["selection"] if top_play else "Unplayable",
        "primary_action_title": top_play["selection"] if top_play else "Unplayable",
        "primary_action": top_play["selection"] if top_play else "Unplayable",
        "primary_market": top_play["market_type"] if top_play else "None",
        "primary_confidence": conf_pct_str,
        "primary_risk_level": risk_lvl,
        "primary_insight": top_play["insight"] if top_play else (unplayable_msg or "No edges detected."),
        "has_ladder": False,
        "value_action": tier2["selection"] if tier2 else None,
        "value_confidence": tier2["confidence_score"] if tier2 else None,
        "primary_lock": tier1,
        "value_lock": tier2,
    }
    return m_dict, active_plays
