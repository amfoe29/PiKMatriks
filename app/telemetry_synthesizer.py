"""Historical Telemetry Synthesizer for Basketball and Tennis.

Synthesizes advanced offensive/defensive metrics and physical condition indices
from historical match records to resolve data starvation:
  - Basketball: ORTG (Offensive Rating), DRTG (Defensive Rating), and Pace.
  - Tennis: Set Dominance (% sets won vs dropped) and Fatigue Index (days since last match).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.models import Entity, Match, MatchStatus, Sport
from app.utils import (
    COMPETITION_TIERS,
    TIER_1_WEIGHT,
    TIER_2_WEIGHT,
    TIER_3_WEIGHT,
    get_competition_tier,
    is_promotion_matchup,
)

log = logging.getLogger("pikmatrix_synthesizer")



def _get_entity_prior_pace(team_name: str) -> dict[str, float]:
    """Retrieve league baseline or team-specific prior ratings."""
    try:
        from app.ml_pipeline import get_basketball_team_pace
        pace_info = get_basketball_team_pace(team_name)
        # Convert avg_points and def_rating into ORTG/DRTG priors on ~100 possessions
        avg_pts = float(pace_info.get("avg_points", 114.0))
        def_rtg = float(pace_info.get("def_rating", 114.0))
        return {
            "ortg": avg_pts,
            "drtg": def_rtg,
            "pace": 99.5,
        }
    except Exception:
        return {
            "ortg": 112.5,
            "drtg": 112.5,
            "pace": 99.5,
        }


def calculate_basketball_metrics(entity_id: int, session: Session) -> dict[str, float]:
    """Query the last 10 finished matches for the team and calculate ORTG, DRTG, and Pace.

    Pace Estimation:
      - Uses box-score FGA, FTA, TOV if available in match metadata or advanced stats.
      - Otherwise, estimates possessions via the standard points/pace formula:
        Possessions ≈ (Team Points + Opponent Points) / 2.24
      - Clamps game pace to realistic professional ranges (88 - 115 poss/game).

    Ratings:
      - ORTG = (Points Scored / Possessions) * 100
      - DRTG = (Points Conceded / Possessions) * 100
      - Pace = Possessions per 48m game
    """
    entity = session.get(Entity, entity_id)
    entity_name = entity.name if entity else f"Team_{entity_id}"
    priors = _get_entity_prior_pace(entity_name)

    stmt = (
        select(Match)
        .where(
            Match.status == MatchStatus.FINISHED,
            or_(Match.home_entity_id == entity_id, Match.away_entity_id == entity_id),
        )
        .order_by(Match.kickoff_time.desc())
        .limit(10)
    )
    finished_matches = session.scalars(stmt).all()

    if not finished_matches:
        return {
            "ortg": round(priors["ortg"], 2),
            "drtg": round(priors["drtg"], 2),
            "pace": round(priors["pace"], 2),
        }

    ortg_list: list[float] = []
    drtg_list: list[float] = []
    pace_list: list[float] = []

    for m in finished_matches:
        is_home = (m.home_entity_id == entity_id)

        # Extract points for and against
        pts_for: Optional[int] = None
        pts_against: Optional[int] = None

        if is_home:
            if m.actual_home_score is not None and m.actual_away_score is not None:
                pts_for = m.actual_home_score
                pts_against = m.actual_away_score
            elif m.final_score:
                pts_for = m.final_score.get("home")
                pts_against = m.final_score.get("away")
        else:
            if m.actual_away_score is not None and m.actual_home_score is not None:
                pts_for = m.actual_away_score
                pts_against = m.actual_home_score
            elif m.final_score:
                pts_for = m.final_score.get("away")
                pts_against = m.final_score.get("home")

        if pts_for is None or pts_against is None or pts_for <= 0:
            continue

        pts_for = float(pts_for)
        pts_against = float(pts_against)

        # Possessions calculation
        possessions: Optional[float] = None

        # Check if match has boxscore stats in metadata
        meta = m.final_score if isinstance(m.final_score, dict) else {}
        fga = meta.get("fga") or meta.get("FGA")
        fta = meta.get("fta") or meta.get("FTA")
        tov = meta.get("tov") or meta.get("TOV")

        if fga is not None and fta is not None and tov is not None:
            try:
                possessions = 0.96 * (float(fga) + 0.44 * float(fta) + float(tov))
            except (ValueError, TypeError):
                possessions = None

        if possessions is None or possessions <= 0:
            # Simplified points/pace formula:
            # Typical basketball match averages ~1.12 points per possession per team (2.24 total)
            total_pts = pts_for + pts_against
            possessions = total_pts / 2.24

        # Clamp possessions to realistic professional boundaries
        possessions = max(88.0, min(115.0, possessions))

        match_ortg = (pts_for / possessions) * 100.0
        match_drtg = (pts_against / possessions) * 100.0

        ortg_list.append(match_ortg)
        drtg_list.append(match_drtg)
        pace_list.append(possessions)

    if not ortg_list:
        return {
            "ortg": round(priors["ortg"], 2),
            "drtg": round(priors["drtg"], 2),
            "pace": round(priors["pace"], 2),
        }

    sample_size = len(ortg_list)
    avg_ortg = sum(ortg_list) / sample_size
    avg_drtg = sum(drtg_list) / sample_size
    avg_pace = sum(pace_list) / sample_size

    # Bayesian shrinkage toward league prior for small sample sizes
    weight_sample = min(1.0, sample_size / 10.0)
    weight_prior = 1.0 - weight_sample

    final_ortg = (avg_ortg * weight_sample) + (priors["ortg"] * weight_prior)
    final_drtg = (avg_drtg * weight_sample) + (priors["drtg"] * weight_prior)
    final_pace = (avg_pace * weight_sample) + (priors["pace"] * weight_prior)

    return {
        "ortg": round(final_ortg, 2),
        "drtg": round(final_drtg, 2),
        "pace": round(final_pace, 2),
    }


def calculate_tennis_metrics(entity_id: int, session: Session) -> dict[str, float]:
    """Query the last 5 matches for the player.

    Calculates:
      - set_dominance: Percentage of sets won vs. dropped across the recent window (0.0 to 100.0%).
      - fatigue_index: Days since the timestamp of their last match.
    """
    stmt = (
        select(Match)
        .where(
            Match.status == MatchStatus.FINISHED,
            or_(Match.home_entity_id == entity_id, Match.away_entity_id == entity_id),
        )
        .order_by(Match.kickoff_time.desc())
        .limit(5)
    )
    matches = session.scalars(stmt).all()

    now_utc = datetime.now(timezone.utc)

    if not matches:
        return {
            "set_dominance": 50.0,
            "fatigue_index": 4.0,  # Standard rested interval (~4 days)
        }

    total_sets_won = 0
    total_sets_lost = 0

    for m in matches:
        is_home = (m.home_entity_id == entity_id)

        # Extract sets won and lost
        sets_w: Optional[int] = None
        sets_l: Optional[int] = None

        if is_home:
            if m.actual_home_score is not None and m.actual_away_score is not None:
                sets_w = m.actual_home_score
                sets_l = m.actual_away_score
            elif m.final_score:
                sets_w = m.final_score.get("home_sets", m.final_score.get("home"))
                sets_l = m.final_score.get("away_sets", m.final_score.get("away"))
        else:
            if m.actual_away_score is not None and m.actual_home_score is not None:
                sets_w = m.actual_away_score
                sets_l = m.actual_home_score
            elif m.final_score:
                sets_w = m.final_score.get("away_sets", m.final_score.get("away"))
                sets_l = m.final_score.get("home_sets", m.final_score.get("home"))

        if sets_w is not None and sets_l is not None:
            try:
                w_int = int(sets_w)
                l_int = int(sets_l)
                if w_int >= 0 and l_int >= 0:
                    total_sets_won += w_int
                    total_sets_lost += l_int
            except (ValueError, TypeError):
                pass

    total_sets = total_sets_won + total_sets_lost
    if total_sets > 0:
        set_dominance = (total_sets_won / total_sets) * 100.0
    else:
        set_dominance = 50.0

    # Calculate fatigue_index: Days since the timestamp of their most recent finished match
    last_match = matches[0]
    last_kickoff = last_match.kickoff_time
    if last_kickoff is not None:
        if last_kickoff.tzinfo is None:
            last_kickoff = last_kickoff.replace(tzinfo=timezone.utc)
        days_diff = (now_utc - last_kickoff).total_seconds() / 86400.0
        fatigue_index = max(0.0, round(days_diff, 2))
    else:
        fatigue_index = 4.0

    return {
        "set_dominance": round(set_dominance, 2),
        "fatigue_index": round(fatigue_index, 2),
    }


def is_mutual_stalemate_match(
    home_entity: Optional[Entity],
    away_entity: Optional[Entity],
) -> tuple[bool, float, dict[str, Any]]:
    """Determine if a matchup meets the Mutual Stalemate condition.

    Condition:
      If both opposing entities concede < 1.0 xGA per match (or < 1.05) and have clean sheet
      rates > 40% (or stalemate_index >= 0.35), flag the matchup with a mutual defensive dampener.

    Returns:
      (is_stalemate, dampener_factor, metadata_dict)
      where dampener_factor is typically 0.82 (in calibrated 0.78x - 0.85x range).
    """
    if not home_entity or not away_entity:
        return False, 1.0, {}

    h_xga = float(getattr(home_entity, "rolling_xga_per_game", 1.2) or 1.2)
    a_xga = float(getattr(away_entity, "rolling_xga_per_game", 1.2) or 1.2)

    h_cs_cnt = getattr(home_entity, "clean_sheets", None)
    a_cs_cnt = getattr(away_entity, "clean_sheets", None)
    h_cs_pct = (h_cs_cnt / 10.0) if h_cs_cnt is not None else (float(getattr(home_entity, "first_half_clean_sheet_pct", 50.0) or 50.0) / 100.0 * 0.7)
    a_cs_pct = (a_cs_cnt / 10.0) if a_cs_cnt is not None else (float(getattr(away_entity, "first_half_clean_sheet_pct", 50.0) or 50.0) / 100.0 * 0.7)

    h_st = float(getattr(home_entity, "stalemate_index", 0.0) or 0.0)
    a_st = float(getattr(away_entity, "stalemate_index", 0.0) or 0.0)

    cond1 = (h_xga < 1.05 and a_xga < 1.05 and h_cs_pct >= 0.38 and a_cs_pct >= 0.38)
    cond2 = (h_st >= 0.35 and a_st >= 0.35 and h_xga < 1.15 and a_xga < 1.15)

    is_stalemate = bool(cond1 or cond2)
    dampener = 0.82 if is_stalemate else 1.0

    meta = {
        "is_mutual_stalemate": is_stalemate,
        "dampener": dampener,
        "home_xga": h_xga,
        "away_xga": a_xga,
        "home_cs_pct": round(h_cs_pct * 100, 1),
        "away_cs_pct": round(a_cs_pct * 100, 1),
        "home_stalemate_index": h_st,
        "away_stalemate_index": a_st,
    }
    return is_stalemate, dampener, meta


def calculate_football_metrics(
    entity_id: int,
    session: Session,
    current_match: Optional[Match] = None,
    target_competition: Optional[str] = None,
) -> dict[str, Any]:
    """Query a football team's last 5-10 finished matches from the database.

    Evaluates match competition_name against the current fixture's tier.
    If a historical match was played in a lower tier (e.g. Championship, Tier 2/3)
    while the current match is Tier 1 (e.g. Premier League, UCL), a Promotion Tax
    is applied to that match:
      - Lower-tier attacking metrics (Goals, xG, Shots) dampened by 0.85x.
      - Lower-tier defensive metrics (Goals Conceded, xGA) inflated by 1.15x.

    Calculates and returns rolling metrics:
      - form_string: e.g. "WWWDW" (W for win, D for draw, L for loss)
      - goals_scored_per_game: average goals scored (taxed if lower tier)
      - goals_conceded_per_game: average goals conceded (taxed if lower tier)
      - over_2_5_pct: percentage of recent matches over 2.5 total goals
      - over_1_5_pct: percentage of recent matches over 1.5 total goals
      - clean_sheets: total clean sheets kept
      - clean_sheet_pct: clean sheet percentage
      - clean_sheet_streak: consecutive clean sheets streak
      - rolling_xga_per_game: rolling Expected Goals Conceded (xGA, taxed if lower tier)
      - rolling_shots_on_target_conceded: rolling shots on target conceded
      - big_chances_ratio: big chances created / conceded ratio
      - first_half_clean_sheet_pct: percentage of matches leading into 0-0 or 0 conceded at HT
      - stalemate_index: rolling rate of low-margin/scoreless outcomes
      - foul_card_intensity: average disciplinary rate per 90
      - promotion_tax_applied: boolean indicating whether lower-tier data was adjusted
    """
    entity = session.get(Entity, entity_id)
    entity_name = entity.name if entity else f"Team_{entity_id}"

    # Determine target competition tier for the current fixture
    target_comp = target_competition or (current_match.competition_name if current_match else None)
    if not target_comp:
        upcoming = session.scalars(
            select(Match)
            .where(
                Match.status.in_([MatchStatus.SCHEDULED, MatchStatus.LIVE, MatchStatus.IN_PLAY]),
                or_(Match.home_entity_id == entity_id, Match.away_entity_id == entity_id),
            )
            .order_by(Match.kickoff_time.asc())
            .limit(1)
        ).first()
        if upcoming and upcoming.competition_name:
            target_comp = upcoming.competition_name

    target_tier, _ = get_competition_tier(target_comp)
    promotion_tax_applied = False

    stmt = (
        select(Match)
        .where(
            Match.status == MatchStatus.FINISHED,
            or_(Match.home_entity_id == entity_id, Match.away_entity_id == entity_id),
        )
        .order_by(Match.kickoff_time.desc())
        .limit(10)
    )
    finished_matches = session.scalars(stmt).all()

    real_results: list[str] = []
    real_gf: list[float] = []
    real_ga: list[float] = []
    real_over_2_5: list[int] = []
    real_over_1_5: list[int] = []
    real_xga: list[float] = []
    real_ht_cs: list[int] = []
    real_stalemates: list[int] = []
    real_cards: list[float] = []
    clean_sheets_count = 0

    attack_str = float(getattr(entity, "attack_strength", 1.2) or 1.2)
    defense_str = float(getattr(entity, "defense_strength", 1.1) or 1.1)

    for m in finished_matches:
        is_home = (m.home_entity_id == entity_id)
        gf: Optional[int] = None
        ga: Optional[int] = None

        if is_home:
            if m.actual_home_score is not None and m.actual_away_score is not None:
                gf, ga = m.actual_home_score, m.actual_away_score
            elif m.home_score is not None and m.away_score is not None:
                gf, ga = m.home_score, m.away_score
            elif m.final_score and isinstance(m.final_score, dict):
                gf = m.final_score.get("home")
                ga = m.final_score.get("away")
        else:
            if m.actual_away_score is not None and m.actual_home_score is not None:
                gf, ga = m.actual_away_score, m.actual_home_score
            elif m.away_score is not None and m.home_score is not None:
                gf, ga = m.away_score, m.home_score
            elif m.final_score and isinstance(m.final_score, dict):
                gf = m.final_score.get("away")
                ga = m.final_score.get("home")

        if gf is not None and ga is not None:
            try:
                gf_val = float(gf)
                ga_val = float(ga)

                # Promotion Tax evaluation:
                # If historical match was in a lower tier (Tier 2/3) but target fixture is Tier 1
                hist_tier, _ = get_competition_tier(m.competition_name)
                is_lower_tier = (target_tier == 1 and hist_tier > 1)
                if is_lower_tier:
                    promotion_tax_applied = True
                    # Dampen lower-tier attacking metrics (Goals, xG, Shots) by 0.85
                    effective_gf = round(gf_val * 0.85, 2)
                    # Inflate lower-tier defensive metrics (Goals Conceded, xGA) by 1.15
                    effective_ga = round(ga_val * 1.15, 2)
                else:
                    effective_gf = gf_val
                    effective_ga = ga_val

                real_gf.append(effective_gf)
                real_ga.append(effective_ga)
                if gf_val > ga_val:
                    real_results.append("W")
                elif gf_val == ga_val:
                    real_results.append("D")
                else:
                    real_results.append("L")

                if ga_val == 0:
                    clean_sheets_count += 1
                if effective_gf + effective_ga > 2.5:
                    real_over_2_5.append(1)
                else:
                    real_over_2_5.append(0)

                if effective_gf + effective_ga > 1.5:
                    real_over_1_5.append(1)
                else:
                    real_over_1_5.append(0)

                # xGA for fixture
                if is_home and m.expected_goals_away is not None:
                    fixture_xga = float(m.expected_goals_away)
                elif not is_home and m.expected_goals_home is not None:
                    fixture_xga = float(m.expected_goals_home)
                else:
                    fixture_xga = round(0.40 * ga_val + 0.60 * max(0.5, 2.2 - defense_str), 2)

                if is_lower_tier:
                    fixture_xga = round(fixture_xga * 1.15, 2)

                real_xga.append(fixture_xga)

                # First half clean sheet
                meta = m.final_score if isinstance(m.final_score, dict) else {}
                ht_co = meta.get("ht_away" if is_home else "ht_home") or meta.get("half_time_away" if is_home else "half_time_home")
                if ht_co is not None:
                    real_ht_cs.append(1 if int(ht_co) == 0 else 0)
                else:
                    if ga_val == 0:
                        real_ht_cs.append(1)
                    elif ga_val <= 1.0:
                        real_ht_cs.append(1 if (len(real_ht_cs) % 2 == 0) else 0)
                    else:
                        real_ht_cs.append(0)

                # Stalemate index (Draws or total goals <= 1)
                if gf_val == ga_val or (effective_gf + effective_ga <= 1):
                    real_stalemates.append(1)
                else:
                    real_stalemates.append(0)

                # Disciplinary / Cards
                cards_val = meta.get("cards") or meta.get("yellow_cards") or meta.get("bookings")
                if cards_val is not None:
                    real_cards.append(float(cards_val))
                else:
                    base_cards = 1.9 + (defense_str - 1.0) * 0.4
                    real_cards.append(round(base_cards, 2))

            except (ValueError, TypeError):
                pass

    # Backfill baseline if fewer than 5 matches in DB
    form_chars = list(real_results)
    if len(form_chars) < 5:
        seed = abs(hash(entity_name))
        ratio = attack_str / max(0.5, defense_str)
        p_w = min(0.65, max(0.25, 0.45 * ratio))
        p_d = 0.28
        base_xga = max(0.65, min(2.10, 2.25 - defense_str))

        while len(form_chars) < 5:
            idx = len(form_chars)
            pseudo_rand = ((seed + idx * 37) % 100) / 100.0
            if pseudo_rand < p_w:
                form_chars.append("W")
                real_gf.append(2.0 * (0.85 if promotion_tax_applied else 1.0))
                real_ga.append(0.0)
                clean_sheets_count += 1
                real_over_1_5.append(1)
                real_over_2_5.append(0)
                real_xga.append(round(max(0.4, base_xga - 0.3) * (1.15 if promotion_tax_applied else 1.0), 2))
                real_ht_cs.append(1)
                real_stalemates.append(0)
                real_cards.append(1.8)
            elif pseudo_rand < p_w + p_d:
                form_chars.append("D")
                real_gf.append(1.0 * (0.85 if promotion_tax_applied else 1.0))
                real_ga.append(1.0 * (1.15 if promotion_tax_applied else 1.0))
                real_over_1_5.append(1)
                real_over_2_5.append(0)
                real_xga.append(round(base_xga * (1.15 if promotion_tax_applied else 1.0), 2))
                real_ht_cs.append(1 if (idx % 2 == 0) else 0)
                real_stalemates.append(1)
                real_cards.append(2.1)
            else:
                form_chars.append("L")
                real_gf.append(0.0)
                real_ga.append(2.0 * (1.15 if promotion_tax_applied else 1.0))
                real_over_1_5.append(1)
                real_over_2_5.append(0)
                real_xga.append(round((base_xga + 0.4) * (1.15 if promotion_tax_applied else 1.0), 2))
                real_ht_cs.append(0)
                real_stalemates.append(0)
                real_cards.append(2.4)

    form_string = "".join(form_chars[:5])
    total_games = len(real_gf) if real_gf else 1
    avg_gf = round(sum(real_gf) / total_games, 2)
    avg_ga = round(sum(real_ga) / total_games, 2)
    over_2_5_pct = round((sum(real_over_2_5) / total_games) * 100.0, 1) if real_over_2_5 else 50.0
    over_1_5_pct = round((sum(real_over_1_5) / total_games) * 100.0, 1) if real_over_1_5 else 75.0
    clean_sheet_pct = round((clean_sheets_count / total_games) * 100.0, 1)

    # Clean sheet streak (from most recent fixture backward)
    clean_sheet_streak = 0
    for ga_val in real_ga:
        if ga_val == 0:
            clean_sheet_streak += 1
        else:
            break

    # Rolling xGA per game
    rolling_xga = round(sum(real_xga) / len(real_xga), 2) if real_xga else round(max(0.65, min(2.10, 2.25 - defense_str)), 2)

    # Shot suppression (shots on target conceded per game)
    rolling_shots_on_target_conceded = round(max(1.8, min(8.0, rolling_xga * 3.3)), 1)

    # Big chances created / conceded ratio
    bc_created = avg_gf + 0.5 * ((getattr(entity, "avg_xg_for", 1.4) or 1.4) * (0.85 if promotion_tax_applied else 1.0))
    bc_conceded = avg_ga + 0.5 * rolling_xga
    big_chances_ratio = round(max(0.40, min(2.80, bc_created / max(0.5, bc_conceded))), 2)

    # First half clean sheet percentage
    first_half_clean_sheet_pct = round((sum(real_ht_cs) / len(real_ht_cs)) * 100.0, 1) if real_ht_cs else 50.0

    # Stalemate index (rate of low-margin/scoreless outcomes)
    stalemate_index = round(sum(real_stalemates) / len(real_stalemates), 2) if real_stalemates else 0.0

    # Disciplinary intensity (average yellow/red card volatility)
    foul_card_intensity = round(sum(real_cards) / len(real_cards), 2) if real_cards else 2.0

    return {
        "form_string": form_string,
        "goals_scored_per_game": avg_gf,
        "goals_conceded_per_game": avg_ga,
        "over_2_5_pct": over_2_5_pct,
        "over_1_5_pct": over_1_5_pct,
        "clean_sheets": clean_sheets_count,
        "clean_sheet_pct": clean_sheet_pct,
        "clean_sheet_streak": clean_sheet_streak,
        "rolling_xga_per_game": rolling_xga,
        "rolling_shots_on_target_conceded": rolling_shots_on_target_conceded,
        "big_chances_ratio": big_chances_ratio,
        "first_half_clean_sheet_pct": first_half_clean_sheet_pct,
        "stalemate_index": stalemate_index,
        "foul_card_intensity": foul_card_intensity,
        "promotion_tax_applied": promotion_tax_applied,
        "target_tier": target_tier,
    }


def synthesize_entity_telemetry(
    entity: Entity,
    session: Session,
    current_match: Optional[Match] = None,
    target_competition: Optional[str] = None,
) -> dict[str, Any]:
    """Synthesizes historical telemetry for an entity and updates the ORM model attributes."""
    sport_slug = (entity.sport.slug if entity.sport else "football").lower()
    updated: dict[str, Any] = {}

    if "basket" in sport_slug:
        b_metrics = calculate_basketball_metrics(entity.id, session)
        entity.ortg = b_metrics["ortg"]
        entity.drtg = b_metrics["drtg"]
        entity.pace = b_metrics["pace"]
        updated = b_metrics
    elif "ten" in sport_slug:
        t_metrics = calculate_tennis_metrics(entity.id, session)
        entity.set_dominance = t_metrics["set_dominance"]
        entity.fatigue_index = t_metrics["fatigue_index"]
        updated = t_metrics
    else:  # Football / Soccer
        f_metrics = calculate_football_metrics(
            entity.id,
            session,
            current_match=current_match,
            target_competition=target_competition,
        )
        entity.form_string = f_metrics["form_string"]
        entity.goals_scored_per_game = f_metrics["goals_scored_per_game"]
        entity.goals_conceded_per_game = f_metrics["goals_conceded_per_game"]
        entity.over_2_5_pct = f_metrics["over_2_5_pct"]
        entity.clean_sheets = f_metrics["clean_sheets"]
        entity.rolling_xga_per_game = f_metrics["rolling_xga_per_game"]
        entity.rolling_shots_on_target_conceded = f_metrics["rolling_shots_on_target_conceded"]
        entity.big_chances_ratio = f_metrics["big_chances_ratio"]
        entity.first_half_clean_sheet_pct = f_metrics["first_half_clean_sheet_pct"]
        entity.stalemate_index = f_metrics["stalemate_index"]
        entity.foul_card_intensity = f_metrics["foul_card_intensity"]
        entity.promotion_tax_applied = bool(f_metrics.get("promotion_tax_applied", False))
        updated = f_metrics

    return updated
