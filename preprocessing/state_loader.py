from datetime import datetime, timezone

from backend.database import SessionLocal
from backend.models import NetworkState


def save_states(
    states: list[dict],
    window_size: int = 5,
) -> int:
    """
    Save temporal network states into PostgreSQL.
    """

    db = SessionLocal()

    try:
        saved_count = 0

        for state in states:

            # Convert Unix timestamp to PostgreSQL-compatible datetime
            timestamp = datetime.fromtimestamp(
                state["start_time"],
                tz=timezone.utc,
            ).replace(tzinfo=None)

            # Calculate network nodes
            # Each unique source/destination IP represents a node.
            nodes = set()

            # Calculate network edges
            # Each unique source -> destination pair represents an edge.
            edges = set()

            # The temporal window currently contains
            # aggregated network sessions.
            # We store the available sessions as state features.
            #
            # Note: generate_time_windows currently returns
            # aggregate statistics rather than the individual
            # sessions, so node/edge information is derived
            # from those statistics where available.

            state_features = {
                "window_index": state["window_index"],
                "sessions": state["sessions"],
                "total_packets": state["total_packets"],
                "total_bytes": state["total_bytes"],
                "unique_sources": state["unique_sources"],
                "unique_destinations": state["unique_destinations"],
                "unique_destination_ports": state[
                    "unique_destination_ports"
                ],
            }

            node_count = (
                state["unique_sources"]
                + state["unique_destinations"]
            )

            edge_count = state["sessions"]

            network_state = NetworkState(
                timestamp=timestamp,
                window_size_seconds=window_size,
                node_count=node_count,
                edge_count=edge_count,
                state_features=state_features,
                anomaly_score=None,
            )

            db.add(network_state)
            saved_count += 1

        db.commit()

        return saved_count

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()