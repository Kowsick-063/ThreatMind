from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


class IncidentModel(Base):
    __tablename__ = "incidents"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    incident_id: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )

    timestamp: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        index=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    risk_level: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )

    peak_attack_probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    peak_horizon: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    attack_category: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    recommended_intervention: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
    )

    decision_basis: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
    )

    mitre_techniques: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
    )

    uncertainties: Mapped[list[str]] = mapped_column(
        JSONB,
        nullable=False,
    )

    evidence: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
    )


class IncidentLifecycleModel(Base):
    __tablename__ = "incident_lifecycle"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    incident_id: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
    )

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )