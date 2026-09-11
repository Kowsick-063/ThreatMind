from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.models import NetworkEvent, NetworkSession

router = APIRouter(
    prefix="/api/network",
    tags=["Network"],
)


@router.get("/sessions")
def get_sessions(
    db: Session = Depends(get_db),
):
    sessions = (
        db.query(NetworkSession)
        .order_by(NetworkSession.start_time.desc())
        .limit(100)
        .all()
    )

    return {
        "count": len(sessions),
        "sessions": [
            {
                "id": session.id,
                "source_ip": session.source_ip,
                "destination_ip": session.destination_ip,
                "source_port": session.source_port,
                "destination_port": session.destination_port,
                "protocol": session.protocol,
                "start_time": session.start_time,
                "end_time": session.end_time,
                "duration": session.duration,
                "packets": session.packets,
                "bytes_transferred": session.bytes_transferred,
                "tcp_flags": session.tcp_flags,
            }
            for session in sessions
        ],
    }


@router.post("/events")
def create_event(
    event_type: str,
    timestamp: datetime,
    source_ip: str | None = None,
    destination_ip: str | None = None,
    session_id: int | None = None,
    db: Session = Depends(get_db),
):
    event = NetworkEvent(
        event_type=event_type,
        timestamp=timestamp,
        source_ip=source_ip,
        destination_ip=destination_ip,
        session_id=session_id,
    )

    db.add(event)
    db.commit()
    db.refresh(event)

    return {
        "message": "Network event created successfully",
        "event_id": event.id,
    }