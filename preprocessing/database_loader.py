from datetime import datetime, timezone

from backend.database import SessionLocal
from backend.models import NetworkSession


def save_sessions(sessions: list[dict]) -> int:
    """
    Save aggregated network sessions into PostgreSQL.
    """

    db = SessionLocal()

    try:
        saved_count = 0

        for session in sessions:
            start_time = datetime.fromtimestamp(
                session["start_time"],
                tz=timezone.utc,
            ).replace(tzinfo=None)

            end_time = datetime.fromtimestamp(
                session["end_time"],
                tz=timezone.utc,
            ).replace(tzinfo=None)

            network_session = NetworkSession(
                source_ip=session["source_ip"],
                destination_ip=session["destination_ip"],
                source_port=session["source_port"],
                destination_port=session["destination_port"],
                protocol=str(session["protocol"]),
                start_time=start_time,
                end_time=end_time,
                duration=session["duration"],
                packets=session["packets"],
                bytes_transferred=session["bytes_transferred"],
            )

            db.add(network_session)
            saved_count += 1

        db.commit()

        return saved_count

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()