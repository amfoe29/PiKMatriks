"""
scraper_test.py - Phase 2 ingestion spike (isolated test: NO database access).

WHY NETWORK INTERCEPTION INSTEAD OF HTML PARSING?
-------------------------------------------------
A modern bookmaker page is a JavaScript app. The HTML you see in DevTools is
rendered *from* JSON that the page's own JS fetches from the bookmaker's
backend API. The pipeline looks like this:

    Browser loads page shell (HTML + JS bundle)
            |
            v
    JS calls backend API  --->  e.g. POST /api/gh/factsCenter/pcEvents
            |                        |
            |                        v
            |                 JSON response: teams, kickoff, markets, odds
            v                        |
    JS renders obfuscated DOM  <-----+   (class names like "m-tb-x9Fq2")
    (what CSS selectors target)

CSS selectors attack the *bottom* of that pipeline, where class names are
randomised on each deploy. We attach a listener to the *middle*: every HTTP
response the browser receives is handed to our Python callback, and we read
the clean, structured JSON before it is ever turned into HTML. The DOM's
obfuscation simply doesn't matter to us.

We still need a real browser (Playwright/Chromium) because the site's JS is
what builds the correct API request (headers, cookies, tokens, request body).
Letting the page do that for us is far more robust than reverse-engineering
and replaying the API calls by hand.

Usage:
    .\\.venv\\Scripts\\python.exe scraper_test.py
    .\\.venv\\Scripts\\python.exe scraper_test.py --headed --dump
    .\\.venv\\Scripts\\python.exe scraper_test.py --url "<another league page>"

stdout  -> ONLY the final JSON array (safe to pipe into a file / jq).
stderr  -> progress logs (what was intercepted, warnings).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence

from playwright.sync_api import BrowserContext, Page, Request, Response, Route
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# SportyBet Ghana league pages. SportyBet uses Sportradar ids in its URLs:
#   sr:category:<n>   = country / circuit  (1 = England, 32 = Spain, 15 = USA)
#   sr:tournament:<n> = competition        (17 = Premier League, 8 = LaLiga, 132 = NBA)
SPORTYBET_BASE = "https://www.sportybet.com/gh/sport"

# International Tournaments (Sportradar sr:category:4 endpoints)
INTERNATIONAL_TOURNAMENT_URLS = {
    # FIFA World Cup & Regional Qualifiers
    "world_cup": f"{SPORTYBET_BASE}/football/sr:category:4/sr:tournament:1",
    "world_cup_qual_uefa": f"{SPORTYBET_BASE}/football/sr:category:4/sr:tournament:369",
    "world_cup_qual_conmebol": f"{SPORTYBET_BASE}/football/sr:category:4/sr:tournament:368",
    "world_cup_qual_caf": f"{SPORTYBET_BASE}/football/sr:category:4/sr:tournament:370",
    "world_cup_qual_afc": f"{SPORTYBET_BASE}/football/sr:category:4/sr:tournament:367",
    "world_cup_qual_concacaf": f"{SPORTYBET_BASE}/football/sr:category:4/sr:tournament:371",
    # UEFA Nations League & UEFA European Championship
    "uefa_nations_league": f"{SPORTYBET_BASE}/football/sr:category:4/sr:tournament:10783",
    "uefa_euro": f"{SPORTYBET_BASE}/football/sr:category:4/sr:tournament:2",
    "uefa_euro_qual": f"{SPORTYBET_BASE}/football/sr:category:4/sr:tournament:365",
    # Africa Cup of Nations (AFCON) & Qualifiers
    "afcon": f"{SPORTYBET_BASE}/football/sr:category:4/sr:tournament:270",
    "afcon_qual": f"{SPORTYBET_BASE}/football/sr:category:4/sr:tournament:372",
    # Copa America
    "copa_america": f"{SPORTYBET_BASE}/football/sr:category:4/sr:tournament:271",
    # International Friendlies
    "international_friendlies": f"{SPORTYBET_BASE}/football/sr:category:4/sr:tournament:685",
}

DEFAULT_URLS = (
    f"{SPORTYBET_BASE}/football/sr:category:1/sr:tournament:17",     # England - Premier League
    f"{SPORTYBET_BASE}/football/sr:category:32/sr:tournament:8",     # Spain - LaLiga
    f"{SPORTYBET_BASE}/basketball/sr:category:15/sr:tournament:132", # USA - NBA
    f"{SPORTYBET_BASE}/tennis",                                      # Tennis
    *INTERNATIONAL_TOURNAMENT_URLS.values(),
)

# A response is only considered if its URL contains one of these substrings.
# SportyBet's event list endpoint is ".../factsCenter/pcEvents", which matches
# "events". This cheap filter discards analytics, config, translations, etc.
URL_KEYWORDS = ("events", "markets", "odds")

# Playwright tags every request with the reason the browser made it.
# "xhr" / "fetch" = data requested by JavaScript - exactly the API calls we want.
# Ignoring "document", "script", "image", "stylesheet" etc. removes most noise.
INTERCEPT_RESOURCE_TYPES = {"xhr", "fetch"}

# Purely visual resources we refuse to download at all (faster, less bandwidth).
# This does not affect the JSON API calls.
BLOCKED_RESOURCE_TYPES = {"image", "media", "font"}

# The goals line we want from the Over/Under market.
TARGET_LINE = 2.5

# How long to keep listening after the page starts loading (seconds).
MAX_WAIT_S = 25
# Stop early once we have data AND no new API payload has arrived for this long.
QUIET_PERIOD_S = 4

# Internal market_type values (match the `bookmaker_odds.market_type` column).
OVER_UNDER = "over_under"
DOUBLE_CHANCE = "double_chance"
HANDICAP = "handicap"
MONEYLINE = "moneyline"


def log(message: str) -> None:
    """Progress messages go to stderr so stdout stays pure JSON."""
    print(f"[scraper] {message}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# 1. NETWORK INTERCEPTION
# --------------------------------------------------------------------------- #
class PayloadCollector:
    """Receives every network response from the page and keeps the JSON ones.

    Playwright emits a "response" event each time the browser receives HTTP
    response headers for ANY request: the HTML document, JS bundles, images,
    tracking pixels, and the API calls we care about. A busy betting page can
    fire hundreds of these, so `handle()` filters aggressively and does as
    little work as possible. Parsing happens later, once the page settles.
    """

    def __init__(self, url_keywords: tuple[str, ...]) -> None:
        self.url_keywords = tuple(k.lower() for k in url_keywords)
        # (url, decoded_json) pairs, in arrival order.
        self.payloads: list[tuple[str, Any]] = []
        # Monotonic timestamp of the last useful capture; used for "quiet" detection.
        self.last_capture_at: float = 0.0

    def _is_candidate(self, response: Response) -> bool:
        """Cheap checks that only use metadata already in memory (no body read)."""
        request: Request = response.request

        # Filter 1: only data requests made by JavaScript (XHR / fetch()).
        if request.resource_type not in INTERCEPT_RESOURCE_TYPES:
            return False

        # Filter 2: URL must look like an events/markets/odds endpoint.
        url = response.url.lower()
        if not any(keyword in url for keyword in self.url_keywords):
            return False

        # Filter 3: successful responses only (2xx). Redirects (3xx) have no body.
        if not response.ok:
            return False

        # Filter 4: server must declare JSON. `response.headers` keys are
        # lower-cased by Playwright, so "content-type" is always the right key.
        return "json" in response.headers.get("content-type", "")

    def handle(self, response: Response) -> None:
        """Callback registered with `page.on("response", ...)`.

        This is the heart of the DOM bypass: we are handed the raw HTTP
        response the page's JavaScript is about to consume, and we decode its
        body ourselves. We never look at the rendered HTML.
        """
        if not self._is_candidate(response):
            return

        try:
            # `.json()` asks Chromium (over the DevTools protocol) for the
            # response body and decodes it. This is the same byte stream the
            # site's JS receives - before any rendering or obfuscation.
            data = response.json()
        except (PlaywrightError, ValueError) as exc:
            # PlaywrightError: body no longer available (e.g. page navigated away).
            # ValueError: header said JSON but the body wasn't valid JSON.
            log(f"skip (unreadable body): {response.url[:100]} -> {exc}")
            return

        self.payloads.append((response.url, data))
        self.last_capture_at = time.monotonic()
        log(f"captured {response.request.method} {response.url[:110]}")


def block_heavy_resources(route: Route) -> None:
    """Request-side interception (the counterpart to response listening).

    `page.route()` lets us decide, *before* a request leaves the browser,
    whether to let it through (`continue_`) or kill it (`abort`). Images,
    fonts and video are irrelevant to odds, so we abort them to load faster.
    """
    if route.request.resource_type in BLOCKED_RESOURCE_TYPES:
        route.abort()
    else:
        route.continue_()


def wait_for_payloads(page: Page, collector: PayloadCollector, max_wait_s: float) -> None:
    """Keep the browser alive while the page's JS performs its API calls.

    IMPORTANT (sync API detail): Playwright only delivers events to our
    `handle()` callback while Python is inside a Playwright call. `time.sleep()`
    would freeze event delivery entirely, so we use `page.wait_for_timeout()`,
    which waits *and* processes incoming network events.
    """
    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        page.wait_for_timeout(500)
        has_data = bool(collector.payloads)
        quiet_for = time.monotonic() - collector.last_capture_at
        if has_data and quiet_for >= QUIET_PERIOD_S:
            log(f"network quiet for {QUIET_PERIOD_S}s - done listening")
            return
    log(f"reached max wait of {max_wait_s}s")


# --------------------------------------------------------------------------- #
# 2. JSON PARSING (schema-tolerant)
# --------------------------------------------------------------------------- #
# Rather than hard-coding one exact JSON path (data[0].events[3].markets...),
# we walk the whole payload and recognise "event-shaped" and "market-shaped"
# objects by their keys. This survives the API wrapping things differently,
# and handles a few common naming styles across bookmakers.

HOME_KEYS = ("homeTeamName", "homeTeam", "home_team", "homeName", "home", "team1")
AWAY_KEYS = ("awayTeamName", "awayTeam", "away_team", "awayName", "away", "team2")
HOME_ID_KEYS = ("homeTeamId", "home_team_id", "homeId", "competitor1Id", "team1Id")
AWAY_ID_KEYS = ("awayTeamId", "away_team_id", "awayId", "competitor2Id", "team2Id")
EVENT_ID_KEYS = ("eventId", "event_id", "matchId", "id")
START_KEYS = (
    "estimateStartTime",  # SportyBet (epoch milliseconds)
    "startTime",
    "start_time",
    "kickoffTime",
    "kickoff",
    "commence_time",
    "scheduled",
    "startDate",
)
OUTCOME_LIST_KEYS = ("outcomes", "selections", "runners")
LABEL_KEYS = ("desc", "name", "title", "label")
PRICE_KEYS = ("odds", "price", "value", "decimal")

# --- Market identification -------------------------------------------------
# Sportradar market ids (SportyBet uses the Sportradar feed). An id match is
# authoritative. Verified against live pcEvents payloads:
# Football:
#   18 = Over/Under Goals (specifier total=2.5)
#   10 = Double Chance (outcomes 9/10/11)
#   14 = 3-way Handicap (specifier hcp=0:1, outcomes 1711/1712/1713)
# Basketball:
#   219 = Match Winner incl. OT (outcomes 4/5)
#   223 = Handicap / Point Spread incl. OT (specifier hcp=-6.5, outcomes 1714/1715)
#   225 = Total Points Over/Under incl. OT (specifier total=...)
# Tennis:
#   186 = Match Winner (outcomes 4/5)
#   187 = Game Handicap (specifier hcp=...)
#   188 = Set Handicap (specifier hcp=...)
#   189 = Total Games Over/Under (specifier total=...)
SPORTRADAR_MARKET_TYPES = {
    # Football
    "10": DOUBLE_CHANCE,
    "14": HANDICAP,
    "18": OVER_UNDER,
    # Basketball
    "219": MONEYLINE,
    "223": HANDICAP,
    "225": OVER_UNDER,
    # Tennis
    "186": MONEYLINE,
    "187": HANDICAP,
    "188": HANDICAP,
    "189": OVER_UNDER,
}

# Fallback for non-Sportradar feeds: EXACT normalised names only. Substring
# matching ("contains over/under") is what let the combo market slip through.
EXACT_MARKET_NAMES = {
    "over/under": OVER_UNDER,
    "double chance": DOUBLE_CHANCE,
    "winner": MONEYLINE,
    "winner (incl. overtime)": MONEYLINE,
    "moneyline": MONEYLINE,
}
HANDICAP_NAME_RE = re.compile(r"(?:game |set )?handicap(?: \(incl\. overtime\))?(?: [+-]?\d+(?::\d+|\.\d+)?)?")

DOUBLE_CHANCE_LABELS = {
    "home or draw": "1X", "1x": "1X", "1 or x": "1X", "1/x": "1X", "home/draw": "1X",
    "home or away": "12", "12": "12", "1 or 2": "12", "1/2": "12", "home/away": "12",
    "draw or away": "X2", "x2": "X2", "x or 2": "X2", "x/2": "X2", "draw/away": "X2",
}
SPORTRADAR_DOUBLE_CHANCE_OUTCOMES = {"9": "1X", "10": "12", "11": "X2"}
SPORTRADAR_HANDICAP_OUTCOMES = {
    "1711": "home", "1712": "draw", "1713": "away",
    "1714": "home", "1715": "away",
}
SPORTRADAR_MONEYLINE_OUTCOMES = {"4": "home", "5": "away"}

# --- Sport / competition identification ------------------------------------
# Sportradar sport ids -> our `sports.slug`.
SPORTRADAR_SPORTS = {"sr:sport:1": "football", "sr:sport:2": "basketball", "sr:sport:5": "tennis"}
SPORT_NAME_ALIASES = {
    "soccer": "football", "football": "football",
    "basketball": "basketball", "tennis": "tennis",
}
COMPETITION_KEYS = (
    "tournamentName", "competitionName", "leagueName", "competition", "league", "tournament",
)

NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")
SPECIFIER_TOTAL_RE = re.compile(r"total=(\d+(?:\.\d+)?)")
# Sportradar 3-way handicap head start "hcp=HOME:AWAY", e.g. "hcp=0:1".
SPECIFIER_HCP_RE = re.compile(r"hcp=(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)")
SPECIFIER_HCP_SINGLE_RE = re.compile(r"hcp=([+-]?\d+(?:\.\d+)?)")
HCP_PAIR_RE = re.compile(r"(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)")
HCP_SINGLE_RE = re.compile(r"\(([+-]?\d+(?:\.\d+)?)\)")


def iter_dicts(node: Any) -> Iterator[dict]:
    """Yield every dict nested anywhere inside a JSON structure (depth-first)."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from iter_dicts(value)
    elif isinstance(node, list):
        for item in node:
            yield from iter_dicts(item)


def first_match(obj: dict, keys: tuple[str, ...], convert: Callable[[Any], Any]) -> Any:
    """Return the first key in `keys` whose value `convert()` turns into non-None."""
    for key in keys:
        if key in obj:
            converted = convert(obj[key])
            if converted is not None:
                return converted
    return None


def to_team_name(value: Any) -> Optional[str]:
    """Team names appear either as plain strings or as {"name": "..."} objects."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        return first_match(value, ("name", "teamName", "title"), to_team_name)
    return None  # ints (e.g. a score {"home": 1}) are correctly ignored


def to_id(value: Any) -> Optional[str]:
    """External ids as strings ("sr:competitor:42", 1234 -> "1234")."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, dict):
        for k in ("id", "competitorId", "externalId", "teamId"):
            if k in value:
                res = to_id(value[k])
                if res:
                    return res
        return None
    if isinstance(value, (str, int)) and str(value).strip():
        return str(value).strip()
    return None


# --------------------------------------------------------------------------- #
# National Teams & International Match Detection
# --------------------------------------------------------------------------- #
KNOWN_NATIONAL_TEAMS = {
    "france", "england", "germany", "spain", "brazil", "argentina", "italy",
    "portugal", "netherlands", "belgium", "croatia", "morocco", "senegal",
    "nigeria", "ghana", "ivory coast", "côte d'ivoire", "cote d'ivoire",
    "cameroon", "egypt", "algeria", "uruguay", "colombia", "usa",
    "united states", "mexico", "japan", "south korea", "korea republic",
    "australia", "denmark", "switzerland", "poland", "scotland", "austria",
    "turkey", "wales", "serbia", "chile", "ecuador", "peru", "paraguay",
    "venezuela", "bolivia", "canada", "panama", "costa rica", "jamaica",
    "tunisia", "mali", "south africa", "congo dr", "dr congo", "zambia",
    "saudi arabia", "iran", "qatar", "uae", "iraq", "uzbekistan", "china",
    "norway", "sweden", "ukraine", "czech republic", "hungary", "greece",
    "romania", "slovakia", "slovenia", "albania", "georgia", "finland",
    "republic of ireland", "northern ireland", "bosnia & herzegovina",
    "iceland", "israel", "montenegro", "north macedonia", "bulgaria",
}

INTERNATIONAL_KEYWORDS = (
    "international", "world cup", "nations league", "euro", "european championship",
    "afcon", "africa cup of nations", "copa america", "friendly", "friendlies",
    "qualifier", "qual.", "fifa", "uefa", "caf", "conmebol", "concacaf", "afc"
)


def is_national_team_name(name: Optional[str]) -> bool:
    if not name:
        return False
    low = name.strip().lower()
    return low in KNOWN_NATIONAL_TEAMS


def is_international_competition(*comp_strings: Any) -> bool:
    for text in comp_strings:
        if not text:
            continue
        low = str(text).lower()
        if any(kw in low for kw in INTERNATIONAL_KEYWORDS):
            return True
    return False


def to_utc_iso(value: Any) -> Optional[str]:
    """Normalise epoch seconds/milliseconds or ISO-8601 strings to UTC ISO-8601."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
        ts = float(value)
        if ts > 1e11:  # anything this large is epoch *milliseconds*
            ts /= 1000
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    return None


def to_price(value: Any) -> Optional[float]:
    """Decimal odds as float. Values <= 1.0 mean suspended/unpriced -> None."""
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if price > 1.0 else None


def label_of(obj: dict) -> str:
    """First non-empty label, normalised (lower-case, single spaces).

    Uses the FIRST label field only: SportyBet markets carry desc, name AND
    title, and joining them would break exact-name matching.
    """
    for key in LABEL_KEYS:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return re.sub(r"\s+", " ", value).strip().lower()
    return ""


def is_active(outcome: dict) -> bool:
    """SportyBet flags suspended/closed outcomes with isActive != 1.

    Feeds without the flag are treated as active.
    """
    return outcome.get("isActive", 1) in (1, True, "1")


def classify_market(market: dict) -> Optional[str]:
    """Map a market object to our market_type, or None if we don't ingest it."""
    market_id = str(market.get("id", ""))
    if market_id in SPORTRADAR_MARKET_TYPES:
        return SPORTRADAR_MARKET_TYPES[market_id]

    name = label_of(market)
    if not name or "&" in name:  # combo markets are never a plain market
        return None
    if name in EXACT_MARKET_NAMES:
        return EXACT_MARKET_NAMES[name]
    if HANDICAP_NAME_RE.fullmatch(name):
        return HANDICAP
    return None


def total_line(market: dict, outcome_label: str) -> Optional[float]:
    """Over/Under line: specifier "total=2.5", then a `line` key, then the label."""
    spec = str(market.get("specifier") or market.get("specifiers") or "")
    if match := SPECIFIER_TOTAL_RE.search(spec):
        return float(match.group(1))
    for key in ("line", "total", "points"):
        try:
            return float(market[key])
        except (KeyError, TypeError, ValueError):
            continue
    if number := NUMBER_RE.search(outcome_label):
        return float(number.group(1))
    return None


def handicap_line(market: dict) -> Optional[float]:
    """Handicap from the HOME side's perspective.

    Sportradar's "hcp=A:B" is a virtual head start (home gets A goals, away
    gets B). Home's handicap is therefore A - B:
        hcp=0:1 -> -1.0  (home must win by 2+ to win the bet)
        hcp=2:0 -> +2.0
    For signed 2-way handicap ("hcp=-6.5" or "hcp=1.5"), the value is used directly.
    """
    spec = str(market.get("specifier") or market.get("specifiers") or "")
    if match := SPECIFIER_HCP_RE.search(spec):
        return float(match.group(1)) - float(match.group(2))
    if match := SPECIFIER_HCP_SINGLE_RE.search(spec):
        return float(match.group(1))
    if match := HCP_PAIR_RE.search(label_of(market)):
        return float(match.group(1)) - float(match.group(2))
    if match := HCP_SINGLE_RE.search(label_of(market)):
        return float(match.group(1))
    for key in ("line", "handicap"):
        try:
            return float(market[key])
        except (KeyError, TypeError, ValueError):
            continue
    return None


def outcome_selection(market_type: str, outcome: dict, label: str) -> Optional[str]:
    """Map an outcome to our normalised `selection` value."""
    outcome_id = str(outcome.get("id", ""))
    if market_type == MONEYLINE:
        if outcome_id in SPORTRADAR_MONEYLINE_OUTCOMES:
            return SPORTRADAR_MONEYLINE_OUTCOMES[outcome_id]
        if label.startswith("home") or label == "1":
            return "home"
        if label.startswith("away") or label == "2":
            return "away"
        return None
    if market_type == OVER_UNDER:
        for side in ("over", "under"):
            if label.startswith(side):
                return side
        return None
    if market_type == DOUBLE_CHANCE:
        return DOUBLE_CHANCE_LABELS.get(label) or SPORTRADAR_DOUBLE_CHANCE_OUTCOMES.get(outcome_id)
    if market_type == HANDICAP:
        for side in ("home", "draw", "away"):  # "Home (0:1)" -> "home"
            if label.startswith(side):
                return side
        return SPORTRADAR_HANDICAP_OUTCOMES.get(outcome_id)
    return None


def extract_odds(event: dict, ou_line: float, sport_slug: Optional[str] = None) -> list[dict]:
    """All ingestible odds lines for one event, as bookmaker_odds-shaped dicts."""
    rows: list[dict] = []
    seen: set[tuple] = set()

    for market in iter_dicts(event):
        outcomes = first_match(
            market, OUTCOME_LIST_KEYS, lambda v: v if isinstance(v, list) else None
        )
        if outcomes is None:
            continue
        market_type = classify_market(market)
        if market_type is None:
            continue

        for outcome in outcomes:
            if not isinstance(outcome, dict) or not is_active(outcome):
                continue
            label = label_of(outcome)
            selection = outcome_selection(market_type, outcome, label)
            if selection is None:
                continue

            if market_type == OVER_UNDER:
                line = total_line(market, label)
                if sport_slug == "football":
                    if line is None or abs(line - ou_line) > 1e-9:
                        continue
                else:
                    if line is None:
                        continue
            elif market_type == HANDICAP:
                line = handicap_line(market)
                if line is None:
                    continue
            else:  # double chance and moneyline have no numeric threshold
                line = None

            price = first_match(outcome, PRICE_KEYS, to_price)
            key = (market_type, selection, line)
            if price is None or key in seen:
                continue
            seen.add(key)
            rows.append(
                {"market_type": market_type, "selection": selection, "line": line, "odds": price}
            )

    return rows


def qualify_competition(category: Any, tournament: Any) -> Optional[str]:
    """'England' + 'Premier League' -> 'England - Premier League'.

    The country/circuit prefix matters: SportyBet lists several different
    "Premier League"s (England, Ghana, Russia, ...).
    """
    tournament_name = to_team_name(tournament)  # accepts str or {"name": ...}
    if not tournament_name:
        return None
    category_name = to_team_name(category)
    name = f"{category_name} - {tournament_name}" if category_name else tournament_name
    return name[:100]  # matches.competition_name is String(100)


def event_sport(event: dict) -> Optional[str]:
    """Sport slug from SportyBet's event["sport"] block (id first, then name)."""
    sport = event.get("sport")
    if isinstance(sport, dict):
        sport_id = str(sport.get("id", ""))
        if sport_id in SPORTRADAR_SPORTS:
            return SPORTRADAR_SPORTS[sport_id]
        sport = sport.get("name")
    if isinstance(sport, str):
        return SPORT_NAME_ALIASES.get(sport.strip().lower())
    return None


def event_competition(event: dict) -> Optional[str]:
    """Competition from event["sport"]["category"]["tournament"] or flat keys."""
    sport = event.get("sport")
    if isinstance(sport, dict) and isinstance(sport.get("category"), dict):
        category = sport["category"]
        if name := qualify_competition(category.get("name"), category.get("tournament")):
            return name
    return first_match(event, COMPETITION_KEYS, lambda v: qualify_competition(None, v))


def iter_dicts_with_context(node: Any, competition: Optional[str] = None) -> Iterator[tuple[dict, Optional[str]]]:
    """Like iter_dicts(), but also carries the nearest enclosing competition.

    Many feeds group events under a tournament object, e.g.
    {"name": "Premier League", "categoryName": "England", "events": [...]},
    and don't repeat the name on every event.
    """
    if isinstance(node, dict):
        if isinstance(node.get("events"), list):
            competition = qualify_competition(node.get("categoryName"), node.get("name")) or competition
        yield node, competition
        for value in node.values():
            yield from iter_dicts_with_context(value, competition)
    elif isinstance(node, list):
        for item in node:
            yield from iter_dicts_with_context(item, competition)


SYNTHETIC_PATTERNS = ("srl", "simulated reality", "cyber", "esports", "electronic")


def is_synthetic(*texts: Any) -> bool:
    """Return True if any provided string contains synthetic/virtual indicator keywords."""
    for text in texts:
        if not text:
            continue
        if isinstance(text, str):
            low = text.lower()
            if any(p in low for p in SYNTHETIC_PATTERNS):
                return True
        elif isinstance(text, dict):
            if is_synthetic(*text.values()):
                return True
        elif isinstance(text, (list, tuple)):
            if is_synthetic(*text):
                return True
    return False


def parse_payloads(payloads: list[Any], ou_line: float = TARGET_LINE) -> list[dict]:
    """Turn raw intercepted JSON into clean, de-duplicated football match records."""
    matches: dict[tuple, dict] = {}

    for item in payloads:
        source_url = item[0]
        data = item[1]
        captured_at = item[2] if len(item) > 2 else None

        for node, inherited_competition in iter_dicts_with_context(data):
            # An "event" is any object that names both a home and away side.
            home = first_match(node, HOME_KEYS, to_team_name)
            away = first_match(node, AWAY_KEYS, to_team_name)
            if not home or not away or home == away:
                continue

            sport_slug = event_sport(node)
            if not sport_slug:
                low_url = source_url.lower()
                if "football" in low_url or "soccer" in low_url:
                    sport_slug = "football"
                elif "basketball" in low_url:
                    sport_slug = "basketball"
                elif "tennis" in low_url:
                    sport_slug = "tennis"

            if not sport_slug:
                sport_slug = "football"

            competition = event_competition(node) or inherited_competition

            category_name = None
            tourn_name = None
            sport_obj = node.get("sport")
            if isinstance(sport_obj, dict):
                cat_obj = sport_obj.get("category")
                if isinstance(cat_obj, dict):
                    category_name = cat_obj.get("name")
                    tourn_obj = cat_obj.get("tournament")
                    if isinstance(tourn_obj, dict):
                        tourn_name = tourn_obj.get("name")
                    elif isinstance(tourn_obj, str):
                        tourn_name = tourn_obj

            # Filter out synthetic / simulated / esports / virtual matches
            if is_synthetic(
                home,
                away,
                competition,
                inherited_competition,
                category_name,
                tourn_name,
                sport_obj,
                node.get("tournament"),
                node.get("category"),
                node.get("tournamentName"),
                node.get("competitionName"),
                node.get("categoryName"),
            ):
                continue

            home_id = first_match(node, HOME_ID_KEYS, to_id)
            if not home_id and isinstance(node.get("homeTeam"), dict):
                home_id = to_id(node["homeTeam"].get("id") or node["homeTeam"].get("competitorId"))

            away_id = first_match(node, AWAY_ID_KEYS, to_id)
            if not away_id and isinstance(node.get("awayTeam"), dict):
                away_id = to_id(node["awayTeam"].get("id") or node["awayTeam"].get("competitorId"))

            is_national = (
                is_international_competition(
                    competition,
                    inherited_competition,
                    category_name,
                    tourn_name,
                    source_url,
                    node.get("categoryName"),
                    node.get("tournamentName"),
                )
                or is_national_team_name(home)
                or is_national_team_name(away)
            )

            # National team fallback identifier: synthesize deterministic nat:<slug> if no standard club ID exists
            if not home_id and is_national:
                clean_home = re.sub(r"[^a-z0-9]+", "_", home.lower()).strip("_")
                home_id = f"nat:{clean_home}"
            if not away_id and is_national:
                clean_away = re.sub(r"[^a-z0-9]+", "_", away.lower()).strip("_")
                away_id = f"nat:{clean_away}"

            record = {
                "event_id": first_match(node, EVENT_ID_KEYS, to_id),
                "sport": sport_slug,
                "competition": competition,
                "match": f"{home} vs {away}",
                "home": home,
                "home_id": home_id,
                "away": away,
                "away_id": away_id,
                "kickoff_utc": first_match(node, START_KEYS, to_utc_iso),
                "source_url": source_url,
                "scraped_at": captured_at,
                "odds": extract_odds(node, ou_line, sport_slug),
                "is_national": is_national,
            }

            # The same fixture often appears in several payloads (list view,
            # live widget, refresh polls). Keep one copy - the richest one.
            key = (record["event_id"],) if record["event_id"] else (
                home.lower(), away.lower(), record["kickoff_utc"])
            existing = matches.get(key)
            if existing is None or len(record["odds"]) > len(existing["odds"]):
                matches[key] = record

    return sorted(
        matches.values(),
        key=lambda r: (r["sport"] or "", r["competition"] or "", r["kickoff_utc"] or "", r["match"]),
    )


# --------------------------------------------------------------------------- #
# 3. ORCHESTRATION
# --------------------------------------------------------------------------- #
def page_slug(url: str) -> str:
    """'.../sport/football/sr:category:1/sr:tournament:17' -> 'football_sr_category_1_sr_tournament_17'."""
    path = url.split("?")[0].split("/sport/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9]+", "_", path).strip("_")[:80] or "page"


def dump_payloads(payloads: list[Any], directory: Path) -> None:
    """Save raw captures to disk - essential when adapting to a new bookmaker.

    The folder is refreshed (old *.json removed) so it always holds one
    consistent snapshot. Nothing is touched if the new capture is empty, so a
    blocked run never wipes a good previous dump.
    """
    if not payloads:
        log(f"nothing captured - keeping existing dump in {directory}")
        return
    directory.mkdir(parents=True, exist_ok=True)
    for old in directory.glob("*.json"):
        old.unlink()
    for index, item in enumerate(payloads, start=1):
        url, data = item[0], item[1]
        slug = re.sub(r"[^A-Za-z0-9]+", "_", url.split("?")[0])[-60:]
        path = directory / f"{index:02d}_{slug}.json"
        path.write_text(
            json.dumps({"url": url, "data": data}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    log(f"dumped {len(payloads)} raw payload(s) to {directory.resolve()}")


def dump_all(payloads_by_page: dict[str, list], root: Path) -> None:
    """One sub-folder per league page, e.g. captures/football_sr_category_1_.../"""
    for page_url, payloads in payloads_by_page.items():
        dump_payloads(payloads, root / page_slug(page_url))


def load_captures(directory: Path) -> list[tuple[str, Any, datetime]]:
    """Reload payloads written by dump_payloads() with their file modification timestamps."""
    if not directory.is_dir():
        raise FileNotFoundError(f"capture directory not found: {directory.resolve()}")
    payloads = []
    for path in sorted(directory.rglob("*.json")):
        wrapped = json.loads(path.read_text(encoding="utf-8"))
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        payloads.append((wrapped["url"], wrapped["data"], mtime))
    log(f"loaded {len(payloads)} captured payload(s) from {directory.resolve()}")
    return payloads


def scrape_page(context: BrowserContext, url: str, max_wait_s: float) -> list[tuple[str, Any, datetime]]:
    """Load ONE page with its own listener, so payloads are attributed to it."""
    collector = PayloadCollector(URL_KEYWORDS)
    page = context.new_page()

    # Request-side: drop images/fonts/media before they're downloaded.
    page.route("**/*", block_heavy_resources)

    # Response-side: THE KEY LINE. Register the listener *before* goto(),
    # otherwise API calls fired during the initial load would be missed.
    page.on("response", collector.handle)

    log(f"opening {url}")
    try:
        # "domcontentloaded" returns as soon as the HTML is parsed; we don't
        # wait for "load" because the API calls are what we care about and
        # wait_for_payloads() handles that explicitly.
        page.goto(url, wait_until="domcontentloaded", timeout=45_000)
    except PlaywrightTimeoutError:
        log("page.goto timed out - continuing with whatever was captured")

    try:
        wait_for_payloads(page, collector, max_wait_s)
    finally:
        page.close()

    now = datetime.now(timezone.utc)
    log(f"{len(collector.payloads)} candidate JSON payload(s) from this page")
    return [(url, data, now) for url, data in collector.payloads]


def scrape(urls: Sequence[str], headed: bool, max_wait_s: float) -> dict[str, list[tuple[str, Any]]]:
    """Visit each league page in turn, sharing one browser context.

    Sharing the context keeps cookies/consent/cache between pages, so later
    pages load faster and look like one normal browsing session.
    """
    results: dict[str, list[tuple[str, Any]]] = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not headed)
        context = browser.new_context(
            locale="en-GB",
            timezone_id="UTC",
            viewport={"width": 1366, "height": 900},
        )
        for url in urls:
            try:
                results[url] = scrape_page(context, url, max_wait_s)
            except PlaywrightError as exc:  # one bad page must not kill the batch
                log(f"failed {url}: {exc}")
                results[url] = []
        browser.close()
    return results


def flatten(payloads_by_page: dict[str, list[tuple[str, Any]]]) -> list[tuple[str, Any]]:
    return [payload for payloads in payloads_by_page.values() for payload in payloads]


def main() -> int:
    # Windows consoles default to cp1252; force UTF-8 so names like "Atlético" print.
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Intercept bookmaker API JSON (test).")
    parser.add_argument("urls", nargs="*", help="League pages to load (default: DEFAULT_URLS).")
    parser.add_argument("--headed", action="store_true", help="Show the browser window.")
    parser.add_argument("--dump", action="store_true", help="Save raw payloads to ./captures/<page>/.")
    parser.add_argument("--from-captures", type=Path, nargs="?", const=Path("captures"), metavar="DIR",
                        help="Parse previously dumped payloads instead of scraping (default DIR: captures).")
    parser.add_argument("--line", type=float, default=TARGET_LINE, help="Over/Under line.")
    parser.add_argument("--max-wait", type=float, default=MAX_WAIT_S, help="Seconds to listen per page.")
    args = parser.parse_args()

    if args.from_captures:
        payloads = load_captures(args.from_captures)
    else:
        payloads_by_page = scrape(args.urls or DEFAULT_URLS, args.headed, args.max_wait)
        if args.dump:
            dump_all(payloads_by_page, Path("captures"))
        payloads = flatten(payloads_by_page)

    matches = parse_payloads(payloads, args.line)
    print(json.dumps(matches, indent=2, ensure_ascii=False))

    if not matches:
        log("no matches parsed. Re-run with --dump --headed and inspect ./captures "
            "to see the real payload shape (or a geo-block / consent page).")
        return 1

    per_competition = Counter((m["sport"], m["competition"]) for m in matches)
    for (sport, competition), count in sorted(per_competition.items(), key=str):
        log(f"  {sport or '?':<11} {competition or '?':<40} {count:>3} match(es)")
    total_lines = sum(len(m["odds"]) for m in matches)
    log(f"{len(matches)} match(es), {total_lines} odds line(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
