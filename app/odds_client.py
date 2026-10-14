"""Bet365 odds client for PikMatrix, backed by The Odds API (the-odds-api.com).

Every price shown in PikMatrix must originate from Bet365. This module is the only
place that talks to the upstream odds provider. It never invents, estimates or
back-solves a price: if Bet365 does not list a market, nothing is stored and the UI
renders "Odds unavailable".

Configuration (environment / .env):
    ODDS_API_KEY            Required. Create one at https://the-odds-api.com
    ODDS_API_TTL_MINUTES    Minimum minutes between refreshes of the same sport (default 120)
    ODDS_API_MAX_SPORTS     Maximum sport keys requested per sync run (default 8)
    ODDS_API_MARKETS        Comma list of featured markets (default h2h,spreads,totals)
    ODDS_API_EVENT_MARKETS  Optional comma list of extra markets fetched per event, for
                            example double_chance. Each event costs extra API credits.
    ODDS_MAX_AGE_HOURS      Prices older than this are ignored by readers (default 24)
    ODDS_MATCH_WINDOW_HOURS Kickoff tolerance when pairing provider events to local
                            matches (default 12)
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import unicodedata
from collections import Counter
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Optional

import httpx

log = logging.getLogger("pikmatrix_odds")

# Label persisted in BookmakerOdds.bookmaker_name and the key used by the provider.
ODDS_BOOKMAKER = "Bet365"
ODDS_API_BOOKMAKER_KEY = "bet365"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ODDS_API_REGION = "uk"

STATE_FILE = Path("data/odds_api_state.json")
_STATE_LOCK = threading.Lock()
_WARNED_NO_KEY = False


# --------------------------------------------------------------------------- #
# Configuration helpers
# --------------------------------------------------------------------------- #

def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, "") or default))
    except ValueError:
        return default


def get_api_key() -> Optional[str]:
    """Return the configured Odds API key, or None when odds sourcing is disabled."""
    key = (os.getenv("ODDS_API_KEY") or "").strip()
    return key or None


def is_configured() -> bool:
    return get_api_key() is not None


def ttl_minutes() -> int:
    return _int_env("ODDS_API_TTL_MINUTES", 120)


def max_sports() -> int:
    return _int_env("ODDS_API_MAX_SPORTS", 8)


def max_age_hours() -> int:
    return _int_env("ODDS_MAX_AGE_HOURS", 24)


def match_window_hours() -> int:
    return _int_env("ODDS_MATCH_WINDOW_HOURS", 12)


def featured_markets() -> str:
    raw = os.getenv("ODDS_API_MARKETS") or "h2h,spreads,totals"
    return ",".join(m.strip() for m in raw.split(",") if m.strip())


def event_markets() -> list[str]:
    raw = os.getenv("ODDS_API_EVENT_MARKETS") or ""
    return [m.strip() for m in raw.split(",") if m.strip()]


def warn_once_no_key() -> None:
    global _WARNED_NO_KEY
    if not _WARNED_NO_KEY:
        _WARNED_NO_KEY = True
        log.warning(
            "ODDS_API_KEY is not set. Bet365 odds are disabled and the UI will show "
            "'Odds unavailable'. Create a key at https://the-odds-api.com and add "
            "ODDS_API_KEY to .env."
        )


# --------------------------------------------------------------------------- #
# Reader helpers (used by the prediction pipeline)
# --------------------------------------------------------------------------- #

def freshness_cutoff() -> datetime:
    """Oldest scraped_at (naive UTC, matching how SQLite stores it) still considered current."""
    return (datetime.now(timezone.utc) - timedelta(hours=max_age_hours())).replace(tzinfo=None)


def bet365_conditions() -> list[Any]:
    """SQLAlchemy WHERE conditions restricting BookmakerOdds to current Bet365 prices."""
    from app.models import BookmakerOdds

    return [
        BookmakerOdds.bookmaker_name == ODDS_BOOKMAKER,
        BookmakerOdds.scraped_at >= freshness_cutoff(),
        BookmakerOdds.odds > 1.0,
    ]


def is_current_bet365(row: Any) -> bool:
    """Python-side equivalent of bet365_conditions() for already-loaded rows."""
    if getattr(row, "bookmaker_name", None) != ODDS_BOOKMAKER:
        return False
    odds = getattr(row, "odds", None)
    if not odds or odds <= 1.0:
        return False
    scraped = getattr(row, "scraped_at", None)
    if scraped is None:
        return False
    if scraped.tzinfo is not None:
        scraped = scraped.astimezone(timezone.utc).replace(tzinfo=None)
    return scraped >= freshness_cutoff()


# --------------------------------------------------------------------------- #
# Persistent state (quota and per-sport refresh times)
# --------------------------------------------------------------------------- #

def _read_state() -> dict[str, Any]:
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception as exc:  # corrupt state must never break a sync
        log.warning("Could not read odds state file: %s", exc)
    return {}


def _write_state(state: dict[str, Any]) -> None:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("Could not write odds state file: %s", exc)


def record_quota(headers: Any) -> None:
    """Persist the provider's quota headers so the UI and operators can see usage."""
    with _STATE_LOCK:
        state = _read_state()
        for header, field in (
            ("x-requests-remaining", "requests_remaining"),
            ("x-requests-used", "requests_used"),
            ("x-requests-last", "requests_last_call"),
        ):
            value = headers.get(header)
            if value is not None:
                try:
                    state[field] = int(float(value))
                except ValueError:
                    pass
        state["quota_updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_state(state)


def mark_fetched(sport_key: str) -> None:
    with _STATE_LOCK:
        state = _read_state()
        state.setdefault("fetched", {})[sport_key] = datetime.now(timezone.utc).isoformat()
        _write_state(state)


def recently_fetched(sport_key: str) -> bool:
    """True if this sport was refreshed within the TTL window (protects API credits)."""
    with _STATE_LOCK:
        stamp = _read_state().get("fetched", {}).get(sport_key)
    if not stamp:
        return False
    try:
        fetched = datetime.fromisoformat(stamp)
    except ValueError:
        return False
    return datetime.now(timezone.utc) - fetched < timedelta(minutes=ttl_minutes())


def get_status() -> dict[str, Any]:
    """Operator-facing summary: is sourcing enabled and how much quota remains."""
    state = _read_state()
    return {
        "bookmaker": ODDS_BOOKMAKER,
        "provider": "the-odds-api.com",
        "enabled": is_configured(),
        "ttl_minutes": ttl_minutes(),
        "requests_remaining": state.get("requests_remaining"),
        "requests_used": state.get("requests_used"),
        "quota_updated_at": state.get("quota_updated_at"),
    }


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #

class OddsApiError(Exception):
    """Raised for provider errors. `fatal` means no further call can succeed this run."""

    def __init__(self, status: int, message: str, fatal: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.fatal = fatal


def _get(path: str, params: dict[str, Any], client: Optional[httpx.Client] = None) -> Any:
    key = get_api_key()
    if not key:
        raise OddsApiError(401, "ODDS_API_KEY is not configured", fatal=True)

    query = dict(params)
    query["apiKey"] = key
    owns_client = client is None
    http = client or httpx.Client(timeout=httpx.Timeout(15.0, connect=10.0))
    try:
        response = http.get(f"{ODDS_API_BASE}{path}", params=query)
    except httpx.HTTPError as exc:
        raise OddsApiError(0, f"network error contacting odds provider: {exc.__class__.__name__}", fatal=True)
    finally:
        if owns_client:
            http.close()

    record_quota(response.headers)

    if response.status_code == 200:
        return response.json()
    if response.status_code == 401:
        raise OddsApiError(401, "Odds provider rejected the API key (401). Check ODDS_API_KEY.", fatal=True)
    if response.status_code == 429:
        raise OddsApiError(429, "Odds provider rate limit or quota reached (429).", fatal=True)
    if response.status_code == 422:
        raise OddsApiError(422, f"Odds provider rejected request parameters for {path} (422).")
    raise OddsApiError(response.status_code, f"Odds provider returned HTTP {response.status_code} for {path}.")


def fetch_active_sports(client: Optional[httpx.Client] = None) -> list[dict[str, Any]]:
    """List in-season sport keys. This endpoint does not consume quota."""
    data = _get("/sports", {}, client)
    return [s for s in data if s.get("active")]


def fetch_sport_odds(sport_key: str, client: Optional[httpx.Client] = None) -> list[dict[str, Any]]:
    """Fetch Bet365 featured-market odds for one sport key."""
    return _get(
        f"/sports/{sport_key}/odds",
        {
            "regions": ODDS_API_REGION,
            "bookmakers": ODDS_API_BOOKMAKER_KEY,
            "markets": featured_markets(),
            "oddsFormat": "decimal",
            "dateFormat": "iso",
        },
        client,
    )


def fetch_event_extra_odds(
    sport_key: str, event_id: str, markets: Iterable[str], client: Optional[httpx.Client] = None
) -> Optional[dict[str, Any]]:
    """Fetch additional markets (for example double_chance) for a single event."""
    try:
        return _get(
            f"/sports/{sport_key}/events/{event_id}/odds",
            {
                "regions": ODDS_API_REGION,
                "bookmakers": ODDS_API_BOOKMAKER_KEY,
                "markets": ",".join(markets),
                "oddsFormat": "decimal",
                "dateFormat": "iso",
            },
            client,
        )
    except OddsApiError as exc:
        if exc.fatal:
            raise
        log.info("Extra markets unavailable for event %s: %s", event_id, exc)
        return None


# --------------------------------------------------------------------------- #
# Name normalisation and event pairing
# --------------------------------------------------------------------------- #

_NAME_NOISE = {
    "fc", "cf", "ac", "as", "sc", "ca", "bc", "afc", "fbc", "ec", "fk", "sk", "sv", "us",
    "ssc", "club", "clube", "de", "da", "do", "the", "cd", "ud", "rc", "rcd", "if",
    "calcio", "futebol", "football",
}


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def normalize_label(text: str) -> str:
    """Lowercase, accent-free, punctuation-free label used for lookups."""
    text = _strip_accents(text or "").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def _name_tokens(name: str) -> list[str]:
    tokens = normalize_label(name).split()
    kept = [t for t in tokens if t not in _NAME_NOISE]
    return kept or tokens


def name_score(a: str, b: str) -> float:
    """Similarity in [0, 1] between two team or player names."""
    ta, tb = _name_tokens(a), _name_tokens(b)
    if not ta or not tb:
        return 0.0
    if ta == tb or set(ta) == set(tb):
        return 1.0
    sa, sb = set(ta), set(tb)
    if sa <= sb or sb <= sa:
        return 0.92
    return SequenceMatcher(None, " ".join(ta), " ".join(tb)).ratio()


NAME_MATCH_THRESHOLD = 0.82


def _parse_commence(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _as_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def find_event_for_match(
    home: str,
    away: str,
    kickoff: datetime,
    events: Iterable[dict[str, Any]],
    allow_swap: bool = False,
) -> Optional[tuple[dict[str, Any], bool]]:
    """Pair a local fixture with a provider event.

    Returns (event, swapped) or None. `swapped` is True when the provider lists the two
    sides in the opposite order (only permitted for individual sports such as tennis).
    A fixture is never paired on a single team name or outside the kickoff window.
    """
    kickoff = _as_utc(kickoff)
    window = timedelta(hours=match_window_hours())
    best: Optional[tuple[float, dict[str, Any], bool]] = None

    for event in events:
        commence = _parse_commence(event.get("commence_time", ""))
        if commence is None or abs(commence - kickoff) > window:
            continue
        e_home, e_away = event.get("home_team", ""), event.get("away_team", "")

        straight = min(name_score(home, e_home), name_score(away, e_away))
        swapped = min(name_score(home, e_away), name_score(away, e_home)) if allow_swap else 0.0

        score, flipped = (straight, False) if straight >= swapped else (swapped, True)
        if score < NAME_MATCH_THRESHOLD:
            continue
        # Prefer the closer kickoff when two events score equally.
        score -= abs((commence - kickoff).total_seconds()) / (window.total_seconds() * 100.0)
        if best is None or score > best[0]:
            best = (score, event, flipped)

    if best is None:
        return None
    return best[1], best[2]


# --------------------------------------------------------------------------- #
# Sport key resolution
# --------------------------------------------------------------------------- #

_SOCCER_COMPETITIONS = {
    "premier league": "soccer_epl",
    "english premier league": "soccer_epl",
    "championship": "soccer_efl_champ",
    "efl championship": "soccer_efl_champ",
    "league one": "soccer_england_league1",
    "league two": "soccer_england_league2",
    "bundesliga": "soccer_germany_bundesliga",
    "2 bundesliga": "soccer_germany_bundesliga2",
    "ligue 1": "soccer_france_ligue_one",
    "ligue 2": "soccer_france_ligue_two",
    "laliga": "soccer_spain_la_liga",
    "la liga": "soccer_spain_la_liga",
    "primera division": "soccer_spain_la_liga",
    "laliga 2": "soccer_spain_segunda_division",
    "segunda division": "soccer_spain_segunda_division",
    "serie a": "soccer_italy_serie_a",
    "serie b": "soccer_italy_serie_b",
    "eredivisie": "soccer_netherlands_eredivisie",
    "primeira liga": "soccer_portugal_primeira_liga",
    "campeonato brasileiro serie a": "soccer_brazil_campeonato",
    "brasileirao": "soccer_brazil_campeonato",
    "uefa champions league": "soccer_uefa_champs_league",
    "champions league": "soccer_uefa_champs_league",
    "uefa europa league": "soccer_uefa_europa_league",
    "europa league": "soccer_uefa_europa_league",
    "uefa europa conference league": "soccer_uefa_europa_conference_league",
    "copa libertadores": "soccer_conmebol_copa_libertadores",
    "copa sudamericana": "soccer_conmebol_copa_sudamericana",
    "mls": "soccer_usa_mls",
    "major league soccer": "soccer_usa_mls",
    "scottish premiership": "soccer_spl",
    "super lig": "soccer_turkey_super_league",
    "liga mx": "soccer_mexico_ligamx",
}

_BASKETBALL_COMPETITIONS = {
    "nba": "basketball_nba",
    "wnba": "basketball_wnba",
    "euroleague": "basketball_euroleague",
    "ncaab": "basketball_ncaab",
    "nbl": "basketball_nbl",
}


def resolve_sport_keys(
    sport_slug: str,
    competitions: Counter,
    active_sports: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """Map local competitions to provider sport keys that are currently in season.

    Returns (sport_keys, unmapped_competitions). Only keys the provider reports as
    active are returned, so no credits are spent on out-of-season leagues.
    """
    active_keys = {s["key"] for s in active_sports}
    wanted: list[str] = []
    unmapped: list[str] = []
    slug = (sport_slug or "").lower()

    if slug == "tennis":
        tennis = [s for s in active_sports if str(s.get("key", "")).startswith("tennis_")]
        for competition in competitions:
            norm = normalize_label(competition)
            tour = "atp" if "atp" in norm else "wta" if "wta" in norm else None
            pool = [s for s in tennis if tour is None or f"tennis_{tour}_" in s["key"]]
            comp_tokens = {t for t in norm.split() if t not in {"atp", "wta", "challenger", "itf"}}
            scored = []
            for s in pool:
                title_tokens = set(normalize_label(s.get("title", "")).split())
                scored.append((len(comp_tokens & title_tokens), s["key"]))
            best = max((sc for sc, _ in scored), default=0)
            picks = [k for sc, k in scored if sc == best and sc > 0]
            if not picks:
                unmapped.append(competition)
            wanted.extend(picks)
    else:
        table = _BASKETBALL_COMPETITIONS if slug == "basketball" else _SOCCER_COMPETITIONS
        for competition in competitions:
            key = table.get(normalize_label(competition))
            if key and key in active_keys:
                wanted.append(key)
            else:
                unmapped.append(competition)

    seen: set[str] = set()
    ordered = [k for k in wanted if not (k in seen or seen.add(k))]
    return ordered, unmapped


# --------------------------------------------------------------------------- #
# Normalisation of provider payloads into BookmakerOdds rows
# --------------------------------------------------------------------------- #

def _same(a: str, b: str) -> bool:
    return normalize_label(a) == normalize_label(b)


def _dc_selection(name: str, home: str, away: str) -> Optional[str]:
    """Interpret a double-chance outcome label."""
    low = normalize_label(name)
    if low in {"1x", "home or draw", "home draw"}:
        return "1X"
    if low in {"x2", "draw or away", "draw away"}:
        return "X2"
    if low in {"12", "home or away", "home away"}:
        return "12"
    has_draw = "draw" in low.split()
    home_norm, away_norm = normalize_label(home), normalize_label(away)
    has_home = bool(home_norm) and home_norm in low
    has_away = bool(away_norm) and away_norm in low
    if has_draw and has_home and not has_away:
        return "1X"
    if has_draw and has_away and not has_home:
        return "X2"
    if has_home and has_away and not has_draw:
        return "12"
    return None


def normalize_event(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert one provider event into BookmakerOdds-shaped rows, Bet365 prices only.

    Output vocabulary matches the rest of the app: market_type in {1x2, moneyline,
    over_under, handicap, double_chance}; selection in {home, draw, away, over, under,
    1X, 12, X2}. Anything the provider does not list is simply absent.
    """
    bookmaker = next(
        (b for b in event.get("bookmakers", []) if b.get("key") == ODDS_API_BOOKMAKER_KEY),
        None,
    )
    if not bookmaker:
        return []

    home, away = event.get("home_team", ""), event.get("away_team", "")
    rows: list[dict[str, Any]] = []

    def add(market_type: str, selection: str, line: Optional[float], price: Any) -> None:
        try:
            odds = float(price)
        except (TypeError, ValueError):
            return
        if odds > 1.0:
            rows.append({"market_type": market_type, "selection": selection, "line": line, "odds": odds})

    for market in bookmaker.get("markets", []):
        key = market.get("key")
        outcomes = market.get("outcomes", [])

        if key == "h2h":
            has_draw = any(_same(o.get("name", ""), "draw") for o in outcomes)
            m_type = "1x2" if has_draw else "moneyline"
            for o in outcomes:
                name = o.get("name", "")
                if _same(name, "draw"):
                    add(m_type, "draw", None, o.get("price"))
                elif _same(name, home):
                    add(m_type, "home", None, o.get("price"))
                elif _same(name, away):
                    add(m_type, "away", None, o.get("price"))

        elif key == "totals":
            for o in outcomes:
                side = normalize_label(o.get("name", ""))
                if side in ("over", "under") and o.get("point") is not None:
                    add("over_under", side, float(o["point"]), o.get("price"))

        elif key == "spreads":
            for o in outcomes:
                if o.get("point") is None:
                    continue
                name = o.get("name", "")
                if _same(name, home):
                    add("handicap", "home", float(o["point"]), o.get("price"))
                elif _same(name, away):
                    add("handicap", "away", float(o["point"]), o.get("price"))

        elif key == "double_chance":
            for o in outcomes:
                selection = _dc_selection(o.get("name", ""), home, away)
                if selection:
                    add("double_chance", selection, None, o.get("price"))

    return rows


def flip_sides(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Swap home and away in rows when the provider lists a fixture in reverse order."""
    swap = {"home": "away", "away": "home", "1X": "X2", "X2": "1X"}
    return [{**r, "selection": swap.get(r["selection"], r["selection"])} for r in rows]
