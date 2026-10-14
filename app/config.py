"""Central Configuration Module for PikMatrix.

Loads environment variables from .env and provides strongly typed settings
for multi-sport data pipelines, external APIs, databases, and application security.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

# Resolve project root and load .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH, override=True)
else:
    load_dotenv()


class Settings:
    """Application settings and API credentials."""

    # ----------------------------------------------------------------------- #
    # Master Multi-Sport API Configuration (API-Sports / RapidAPI)
    # ----------------------------------------------------------------------- #
    API_SPORTS_KEY: str = os.getenv(
        "API_SPORTS_KEY",
        "11de7e79f0c61af40cff0d4a4d22a266c8a83e05a39bfa3440a55792556c0b7d",
    ).strip()

    # Preferred header: 'x-apisports-key' or 'x-rapidapi-key'
    API_SPORTS_HEADER: str = os.getenv("API_SPORTS_HEADER", "x-apisports-key").strip()

    # Multi-Sport Base Endpoints
    FOOTBALL_API_URL: str = os.getenv(
        "FOOTBALL_API_URL", "https://v3.football.api-sports.io/fixtures"
    ).strip()
    BASKETBALL_API_URL: str = os.getenv(
        "BASKETBALL_API_URL", "https://v1.basketball.api-sports.io/games"
    ).strip()
    TENNIS_API_URL: str = os.getenv(
        "TENNIS_API_URL", "https://v2.tennis.api-sports.io/matches"
    ).strip()

    # Sport to URL Mapping
    SPORT_ENDPOINTS: Dict[str, str] = {
        "football": FOOTBALL_API_URL,
        "soccer": FOOTBALL_API_URL,
        "basketball": BASKETBALL_API_URL,
        "tennis": TENNIS_API_URL,
    }

    # ----------------------------------------------------------------------- #
    # Legacy / Alternate APIs
    # ----------------------------------------------------------------------- #
    RAPIDAPI_KEY: str = os.getenv("RAPIDAPI_KEY", "").strip()
    ODDS_API_KEY: str = os.getenv("ODDS_API_KEY", "").strip()
    FOOTBALL_DATA_API_KEY: str = os.getenv("FOOTBALL_DATA_API_KEY", "41a2be078da14c399ffc37df5a111be9").strip()
    BALLDONTLIE_API_KEY: str = os.getenv("BALLDONTLIE_API_KEY", "").strip()

    # ----------------------------------------------------------------------- #
    # Database & Security
    # ----------------------------------------------------------------------- #
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./pikmatrix.db").strip()
    DB_ECHO: bool = os.getenv("DB_ECHO", "false").lower() in ("true", "1", "yes")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "pikmatrix-secret-key-change-in-production-2026").strip()
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
    ADMIN_API_KEY: str = os.getenv("ADMIN_API_KEY", "pikmatrix-admin-2026").strip()

    # ----------------------------------------------------------------------- #
    # Header Helper
    # ----------------------------------------------------------------------- #
    def get_api_sports_headers(
        self,
        header_name: Optional[str] = None,
        include_rapidapi_fallback: bool = True,
    ) -> Dict[str, str]:
        """Construct authentication headers for API-Sports requests.

        Args:
            header_name: Specific header to set ('x-apisports-key' or 'x-rapidapi-key').
            include_rapidapi_fallback: Whether to populate both headers for maximum gateway compatibility.

        Returns:
            Dictionary of request headers.
        """
        primary_header = (header_name or self.API_SPORTS_HEADER).lower()
        headers: Dict[str, str] = {
            "Accept": "application/json",
            "User-Agent": "PikMatrix-Pipeline/2.0 (Multi-Sport Data Ingestion)",
        }

        if primary_header == "x-rapidapi-key":
            headers["x-rapidapi-key"] = self.API_SPORTS_KEY
            if include_rapidapi_fallback:
                headers["x-apisports-key"] = self.API_SPORTS_KEY
        else:
            headers["x-apisports-key"] = self.API_SPORTS_KEY
            if include_rapidapi_fallback:
                headers["x-rapidapi-key"] = self.API_SPORTS_KEY

        return headers

    def get_endpoint_for_sport(self, sport: str) -> str:
        """Resolve the target API-Sports endpoint based on sport slug.

        Args:
            sport: Sport identifier (e.g. 'football', 'basketball', 'tennis').

        Returns:
            Base URL string for the sport's fixtures/games/matches endpoint.
        """
        clean_sport = (sport or "").strip().lower()
        if clean_sport in self.SPORT_ENDPOINTS:
            return self.SPORT_ENDPOINTS[clean_sport]
        raise ValueError(
            f"Unsupported sport: '{sport}'. Supported sports: {list(self.SPORT_ENDPOINTS.keys())}"
        )


# Export singleton instance and convenience constants
settings = Settings()
API_SPORTS_KEY = settings.API_SPORTS_KEY
FOOTBALL_API_URL = settings.FOOTBALL_API_URL
BASKETBALL_API_URL = settings.BASKETBALL_API_URL
TENNIS_API_URL = settings.TENNIS_API_URL
