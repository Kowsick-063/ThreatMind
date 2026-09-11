from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


class NetworkSession(Base):
    __tablename__ = "network_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    source_ip: Mapped[str] = mapped_column(String(45), nullable=False)
    destination_ip: Mapped[str] = mapped_column(String(45), nullable=False)

    source_port: Mapped[int | None] = mapped_column(Integer)
    destination_port: Mapped[int | None] = mapped_column(Integer)

    protocol: Mapped[str | None] = mapped_column(String(20))

    start_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
    )

    end_time: Mapped[datetime | None] = mapped_column(DateTime)

    duration: Mapped[float | None] = mapped_column(Float)

    packets: Mapped[int | None] = mapped_column(Integer)
    bytes_transferred: Mapped[int | None] = mapped_column(Integer)

    tcp_flags: Mapped[str | None] = mapped_column(String(50))

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )


class NetworkEvent(Base):
    __tablename__ = "network_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    session_id: Mapped[int | None] = mapped_column(
        ForeignKey("network_sessions.id")
    )

    event_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
    )

    source_ip: Mapped[str | None] = mapped_column(String(45))
    destination_ip: Mapped[str | None] = mapped_column(String(45))

    features: Mapped[dict | None] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )


class NetworkState(Base):
    __tablename__ = "network_states"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    timestamp: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
    )

    window_size_seconds: Mapped[int] = mapped_column(
        Integer,
        default=5,
        nullable=False,
    )

    node_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )

    edge_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )

    state_features: Mapped[dict | None] = mapped_column(JSONB)

    anomaly_score: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    state_id: Mapped[int | None] = mapped_column(
        ForeignKey("network_states.id")
    )

    prediction_time: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
    )

    horizon: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    predicted_class: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    model_name: Mapped[str | None] = mapped_column(
        String(100)
    )

    model_version: Mapped[str | None] = mapped_column(
        String(50)
    )

    explanation: Mapped[dict | None] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )


class AttackTrajectory(Base):
    __tablename__ = "attack_trajectories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    prediction_id: Mapped[int | None] = mapped_column(
        ForeignKey("predictions.id")
    )

    trajectory_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    trajectory: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
    )

    attack_stage: Mapped[str | None] = mapped_column(
        String(100)
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )


class Intervention(Base):
    __tablename__ = "interventions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    intervention_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    target: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    operational_cost: Mapped[float] = mapped_column(
        Float,
        default=0,
    )

    service_disruption: Mapped[float] = mapped_column(
        Float,
        default=0,
    )

    description: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )


class SimulationResult(Base):
    __tablename__ = "simulation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    intervention_id: Mapped[int] = mapped_column(
        ForeignKey("interventions.id"),
        nullable=False,
    )

    baseline_risk: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    simulated_risk: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    risk_reduction: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    defense_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    simulated_state: Mapped[dict | None] = mapped_column(JSONB)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False,
    )