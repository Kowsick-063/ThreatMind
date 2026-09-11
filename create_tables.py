from backend.database import Base, engine
from backend.models import (
    NetworkSession,
    NetworkEvent,
    NetworkState,
    Prediction,
    AttackTrajectory,
    Intervention,
    SimulationResult,
)

print("Creating ThreatMind database tables...")

Base.metadata.create_all(bind=engine)

print("Database tables created successfully!")