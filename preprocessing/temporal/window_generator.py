from collections import defaultdict


def generate_time_windows(
    sessions: list[dict],
    window_size: int = 5,
) -> list[dict]:
    """
    Convert network sessions into fixed-size temporal windows.

    Parameters:
        sessions: Aggregated network sessions.
        window_size: Window duration in seconds.

    Returns:
        List of temporal network states.
    """

    if not sessions:
        return []

    # Find the earliest session start time
    start_time = min(
        session["start_time"]
        for session in sessions
    )

    windows = defaultdict(list)

    for session in sessions:

        # Determine which window this session belongs to
        window_index = int(
            (session["start_time"] - start_time)
            // window_size
        )

        windows[window_index].append(session)

    states = []

    for window_index in sorted(windows):

        window_sessions = windows[window_index]

        total_packets = sum(
            session["packets"] or 0
            for session in window_sessions
        )

        total_bytes = sum(
            session["bytes_transferred"] or 0
            for session in window_sessions
        )

        unique_sources = len({
            session["source_ip"]
            for session in window_sessions
        })

        unique_destinations = len({
            session["destination_ip"]
            for session in window_sessions
        })

        unique_ports = len({
            session["destination_port"]
            for session in window_sessions
            if session["destination_port"] is not None
        })

        states.append(
            {
                "window_index": window_index,
                "start_time": start_time
                + window_index * window_size,
                "end_time": start_time
                + (window_index + 1) * window_size,
                "sessions": len(window_sessions),
                "total_packets": total_packets,
                "total_bytes": total_bytes,
                "unique_sources": unique_sources,
                "unique_destinations": unique_destinations,
                "unique_destination_ports": unique_ports,
            }
        )

    return states