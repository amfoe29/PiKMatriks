"""ORM models for the PikMatrix relational schema."""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, List, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base

# JSONB on PostgreSQL (indexable, binary); plain JSON on SQLite fallback.
JSONType = JSON().with_variant(JSONB(), "postgresql")


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class MatchStatus(str, enum.Enum):
    SCHEDULED = "scheduled"
    LIVE = "live"
    FINISHED = "finished"
    CANCELLED = "cancelled"


class MarketType(str, enum.Enum):
    """Known markets. Stored as plain strings so new markets need no migration."""

    MATCH_RESULT = "1x2"
    DOUBLE_CHANCE = "double_chance"
    OVER_UNDER = "over_under"
    OVER_UNDER_GOALS = "over_under_goals"
    OVER_UNDER_CORNERS = "over_under_corners"
    HANDICAP = "handicap"


class Selection(str, enum.Enum):
    """Common selections. Stored as plain strings for flexibility."""

    HOME = "home"
    AWAY = "away"
    DRAW = "draw"
    HOME_OR_DRAW = "1X"
    AWAY_OR_DRAW = "X2"
    HOME_OR_AWAY = "12"
    OVER = "over"
    UNDER = "under"


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class Sport(Base):
    __tablename__ = "sports"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50), unique=True)
    slug: Mapped[str] = mapped_column(String(50), unique=True, index=True)

    entities: Mapped[List[Entity]] = relationship(back_populates="sport")
    matches: Mapped[List[Match]] = relationship(back_populates="sport")

    def __repr__(self) -> str:
        return f"<Sport {self.slug}>"


class Entity(Base):
    """A team or individual competitor (e.g. 'Arsenal', 'Carlos Alcaraz')."""

    __tablename__ = "entities"
    __table_args__ = (
        UniqueConstraint("sport_id", "name", name="uq_entities_sport_name"),
        # Target for the composite FKs on matches (enforces same-sport fixtures).
        UniqueConstraint("sport_id", "id", name="uq_entities_sport_id_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    sport_id: Mapped[int] = mapped_column(
        ForeignKey("sports.id", ondelete="RESTRICT"), index=True
    )
    name: Mapped[str] = mapped_column(String(150))
    is_individual: Mapped[bool] = mapped_column(Boolean, default=False)
    # `metadata` is reserved by SQLAlchemy's declarative API, so the Python
    # attribute is `entity_metadata` while the DB column stays `metadata`.
    entity_metadata: Mapped[Optional[dict[str, Any]]] = mapped_column(
        "metadata", JSONType
    )
    logo_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)

    # Aggregate and spatial/statistical telemetry
    average_xg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    average_possession: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    attack_strength: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=0.0)
    defense_strength: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=0.0)
    avg_xg_for: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=0.0)
    avg_xg_against: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=0.0)
    avg_corners: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=0.0)
    form_rating: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=0.0)

    # Advanced synthesized telemetry for Basketball & Tennis
    ortg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Offensive Rating (Points per 100 possessions)
    drtg: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Defensive Rating
    pace: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Estimated possessions per game
    set_dominance: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Tennis rolling set win percentage
    fatigue_index: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Days since last match

    # Rolling historical data for Football telemetry & comparison
    form_string: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # e.g., "WWWDW"
    goals_scored_per_game: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    goals_conceded_per_game: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    over_2_5_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # % of recent games > 2.5 goals
    clean_sheets: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Advanced statistical telemetry & defensive suppression
    rolling_xga_per_game: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=1.2)  # Rolling Expected Goals Conceded
    rolling_shots_on_target_conceded: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=4.0)
    big_chances_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=1.0)  # Big chances created / conceded ratio
    first_half_clean_sheet_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=50.0)  # % of matches leading into 0-0 at HT
    stalemate_index: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=0.0)  # Rolling rate of low-margin/scoreless outcomes
    foul_card_intensity: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=2.0)  # Average disciplinary rate per 90
    promotion_tax_applied: Mapped[Optional[bool]] = mapped_column(Boolean, default=False, server_default="0", nullable=True)  # True if promotion tax applied

    sport: Mapped[Sport] = relationship(back_populates="entities")
    home_matches: Mapped[List[Match]] = relationship(
        back_populates="home_entity",
        primaryjoin="Entity.id == Match.home_entity_id",
        foreign_keys="Match.home_entity_id",
    )
    away_matches: Mapped[List[Match]] = relationship(
        back_populates="away_entity",
        primaryjoin="Entity.id == Match.away_entity_id",
        foreign_keys="Match.away_entity_id",
    )

    def __repr__(self) -> str:
        return f"<Entity {self.name!r}>"


class Match(Base):
    __tablename__ = "matches"
    __table_args__ = (
        CheckConstraint(
            "home_entity_id <> away_entity_id", name="ck_matches_distinct_entities"
        ),
        # Composite FKs: both competitors must belong to the match's sport.
        ForeignKeyConstraint(
            ["sport_id", "home_entity_id"],
            ["entities.sport_id", "entities.id"],
            ondelete="RESTRICT",
            name="fk_matches_home_entity_same_sport",
        ),
        ForeignKeyConstraint(
            ["sport_id", "away_entity_id"],
            ["entities.sport_id", "entities.id"],
            ondelete="RESTRICT",
            name="fk_matches_away_entity_same_sport",
        ),
        Index("ix_matches_sport_kickoff", "sport_id", "kickoff_time"),
        # Lookup key used by ingestion to find an existing fixture.
        Index(
            "ix_matches_identity",
            "sport_id",
            "competition_name",
            "home_entity_id",
            "away_entity_id",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    sport_id: Mapped[int] = mapped_column(ForeignKey("sports.id", ondelete="RESTRICT"))
    # e.g. "England - Premier League", "England - FA Cup". Lets models tell
    # league fixtures from cup ties between the same two sides.
    competition_name: Mapped[str] = mapped_column(String(100))
    home_entity_id: Mapped[int] = mapped_column(index=True)
    away_entity_id: Mapped[int] = mapped_column(index=True)
    kickoff_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    status: Mapped[MatchStatus] = mapped_column(
        Enum(
            MatchStatus,
            name="match_status",
            values_callable=lambda e: [m.value for m in e],  # store lowercase values
            validate_strings=True,
        ),
        default=MatchStatus.SCHEDULED,
        server_default=MatchStatus.SCHEDULED.value,
        index=True,
    )
    final_score: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONType)
    # Advanced quantitative telemetry
    expected_goals_home: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    expected_goals_away: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    corner_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    expected_total_goals: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    home_win_prob_baseline: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    away_win_prob_baseline: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Automated Prediction Grading & Verification
    actual_home_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    actual_away_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    prediction_outcome_1x2: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # "CORRECT", "INCORRECT", "VOID"
    prediction_exact_score: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)
    is_graded: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0", index=True)

    sport: Mapped[Sport] = relationship(back_populates="matches")
    home_entity: Mapped[Entity] = relationship(
        back_populates="home_matches",
        primaryjoin="Entity.id == Match.home_entity_id",
        foreign_keys=[home_entity_id],
    )
    away_entity: Mapped[Entity] = relationship(
        back_populates="away_matches",
        primaryjoin="Entity.id == Match.away_entity_id",
        foreign_keys=[away_entity_id],
    )
    odds: Mapped[List[BookmakerOdds]] = relationship(
        back_populates="match", cascade="all, delete-orphan", passive_deletes=True
    )
    predictions: Mapped[List[AIPrediction]] = relationship(
        back_populates="match", cascade="all, delete-orphan", passive_deletes=True
    )
    team_advanced_stats: Mapped[List[TeamAdvancedStats]] = relationship(
        back_populates="match", cascade="all, delete-orphan", passive_deletes=True
    )
    player_match_stats: Mapped[List[PlayerMatchStats]] = relationship(
        back_populates="match", cascade="all, delete-orphan", passive_deletes=True
    )

    @property
    def ai_confidence_pct(self) -> int:
        """Dynamic AI confidence percentage generated by Bivariate Poisson or baseline."""
        if self.home_win_prob_baseline:
            base = max(self.home_win_prob_baseline, self.away_win_prob_baseline or 0.0)
            return max(58, min(94, int(round(base * 100))))
        return 75

    def __repr__(self) -> str:
        return f"<Match {self.id} {self.home_entity_id} v {self.away_entity_id}>"


class BookmakerOdds(Base):
    """A single scraped price snapshot for one selection of one market."""

    __tablename__ = "bookmaker_odds"
    __table_args__ = (
        CheckConstraint("odds > 1.0", name="ck_bookmaker_odds_odds_gt_1"),
        Index(
            "ix_bookmaker_odds_lookup",
            "match_id",
            "market_type",
            "selection",
            "line",
            "bookmaker_name",
            "scraped_at",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"))
    bookmaker_name: Mapped[str] = mapped_column(String(100))
    market_type: Mapped[str] = mapped_column(String(50))
    selection: Mapped[str] = mapped_column(String(50))
    line: Mapped[Optional[float]] = mapped_column(Float)  # NULL for 1X2 / double chance
    odds: Mapped[float] = mapped_column(Float)
    scraped_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    match: Mapped[Match] = relationship(back_populates="odds")

    def __repr__(self) -> str:
        return (
            f"<Odds {self.bookmaker_name} {self.market_type}:"
            f"{self.selection}@{self.line} = {self.odds}>"
        )


class AIPrediction(Base):
    """A model's probability estimate for one selection of one market."""

    __tablename__ = "ai_predictions"
    __table_args__ = (
        CheckConstraint(
            "predicted_probability >= 0 AND predicted_probability <= 1",
            name="ck_ai_predictions_probability_range",
        ),
        CheckConstraint(
            "confidence_score >= 0 AND confidence_score <= 1",
            name="ck_ai_predictions_confidence_range",
        ),
        Index(
            "ix_ai_predictions_lookup", "match_id", "market_type", "selection", "line"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(ForeignKey("matches.id", ondelete="CASCADE"))
    model_name: Mapped[str] = mapped_column(String(100), index=True)
    market_type: Mapped[str] = mapped_column(String(100))
    selection: Mapped[str] = mapped_column(String(150))
    line: Mapped[Optional[float]] = mapped_column(Float)
    predicted_probability: Mapped[float] = mapped_column(Float)
    confidence_score: Mapped[float] = mapped_column(Float)
    is_safe_pick: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    ai_verdict: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    is_alternate: Mapped[bool] = mapped_column(Boolean, default=False, nullable=True)
    cushion_tag: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    value_line: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    value_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    settlement_status: Mapped[str] = mapped_column(
        String(20), default="PENDING", server_default="PENDING", index=True
    )
    settled_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    actual_result_context: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    match: Mapped[Match] = relationship(back_populates="predictions")

    def __repr__(self) -> str:
        return (
            f"<Prediction {self.model_name} {self.market_type}:"
            f"{self.selection}@{self.line} p={self.predicted_probability}>"
        )


class TeamAdvancedStats(Base):
    """Deep statistical telemetry for teams linked to Match and Entity."""

    __tablename__ = "team_advanced_stats"
    __table_args__ = (
        Index("ix_team_adv_stats_entity_date", "entity_id", "recorded_at"),
        Index("ix_team_adv_stats_match_entity", "match_id", "entity_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("matches.id", ondelete="CASCADE"), index=True, nullable=True
    )
    entity_id: Mapped[int] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), index=True
    )

    # Football Telemetry
    npxg_for: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    npxg_against: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    field_tilt_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Basketball Telemetry
    off_rating: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    def_rating: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pace: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Tennis Telemetry
    surface_hold_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    break_points_saved_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    match: Mapped[Optional[Match]] = relationship(back_populates="team_advanced_stats")
    entity: Mapped[Entity] = relationship()

    def __repr__(self) -> str:
        return f"<TeamAdvancedStats entity={self.entity_id} match={self.match_id}>"


class PlayerMatchStats(Base):
    """Player-level match performance and expected metrics."""

    __tablename__ = "player_match_stats"
    __table_args__ = (
        Index("ix_player_stats_match_entity", "match_id", "entity_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    match_id: Mapped[int] = mapped_column(
        ForeignKey("matches.id", ondelete="CASCADE"), index=True
    )
    entity_id: Mapped[int] = mapped_column(
        ForeignKey("entities.id", ondelete="CASCADE"), index=True
    )

    minutes_played: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    usage_rate: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    xg_individual: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    match: Mapped[Match] = relationship(back_populates="player_match_stats")
    entity: Mapped[Entity] = relationship()

    def __repr__(self) -> str:
        return (
            f"<PlayerMatchStats entity={self.entity_id} match={self.match_id} "
            f"mins={self.minutes_played} xg={self.xg_individual}>"
        )


class User(Base):
    """User account model for authentication and session management."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    username: Mapped[str] = mapped_column(String(100), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email} username={self.username}>"


