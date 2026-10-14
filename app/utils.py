"""Utility constants and helper functions for competition tier classifications."""
from __future__ import annotations

from typing import Optional

# Strict League Tier Weight System
# Tier 1 (Premier League, UCL, La Liga, Serie A, Bundesliga, Ligue 1) = 1.0
# Tier 2 (Championship, Segunda Division, Serie B, 2. Bundesliga, Ligue 2, Eredivisie) = 0.85
# Tier 3 (League One, League Two, 3. Liga, National League, and lower leagues) = 0.75
COMPETITION_TIERS = {
    1: 1.0,
    2: 0.85,
    3: 0.75,
}

TIER_1_WEIGHT = 1.0
TIER_2_WEIGHT = 0.85
TIER_3_WEIGHT = 0.75


def get_competition_tier(competition_name: Optional[str]) -> tuple[int, float]:
    """Classify competition into Tier 1, 2, or 3 with standard tier weight coefficient.

    Returns:
        (tier_number, tier_weight)
        Tier 1: (1, 1.0)  - Premier League, UCL, La Liga, Serie A, Bundesliga, Ligue 1
        Tier 2: (2, 0.85) - Championship, Serie B, 2. Bundesliga, Segunda, Ligue 2
        Tier 3: (3, 0.75) - League One, League Two, lower divisions / regional
    """
    if not competition_name:
        return 1, 1.0

    name_clean = competition_name.strip().lower()

    # Tier 2 specific matches (check before Tier 1 to avoid 'Bundesliga' catching '2. Bundesliga')
    if any(k in name_clean for k in [
        "2. bundesliga", "2.bundesliga", "serie b", "segunda", "laliga 2", "hypermotion",
        "championship", "ligue 2", "usl championship", "eredivisie", "primeira liga",
        "super lig", "division 2", "second division",
    ]):
        return 2, 0.85

    # Tier 3 specific matches
    if any(k in name_clean for k in [
        "league one", "league two", "3. liga", "serie c", "national league", "conference",
        "division 3", "tier 3", "segunda b", "primera federacion", "third division",
    ]):
        return 3, 0.75

    # Tier 1 matches
    if any(k in name_clean for k in [
        "premier league", "champions league", "ucl", "la liga", "primera division",
        "serie a", "bundesliga", "ligue 1", "copa libertadores", "europa league",
        "division 1", "first division",
    ]):
        return 1, 1.0

    # Default to Tier 1 for top-flight schedules unless classified as lower tier
    return 1, 1.0


def is_promotion_matchup(historical_comp: Optional[str], current_comp: Optional[str]) -> bool:
    """Determine whether historical match data was played in a lower tier while current match is Tier 1.

    Triggers the 'Promotion Tax' to prevent lower-tier telemetry from bleeding into top-flight projections.
    """
    target_tier, _ = get_competition_tier(current_comp)
    hist_tier, _ = get_competition_tier(historical_comp)

    # True if historical match was played in a lower tier (Tier 2 or 3) and current is Tier 1
    return target_tier == 1 and hist_tier > 1
