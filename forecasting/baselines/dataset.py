FEATURE_COLUMNS = [
    "sessions",
    "total_packets",
    "total_bytes",
    "unique_sources",
    "unique_destinations",
    "unique_destination_ports",
    "node_count",
    "edge_count",
]


def build_transition_dataset(states: list[dict]):
    """
    Convert sequential network states into
    input-target transition pairs.

    S(t) -> S(t+1)
    """

    if len(states) < 2:
        return [], []

    X = []
    y = []

    for current, next_state in zip(states, states[1:]):

        current_features = [
            current["sessions"],
            current["total_packets"],
            current["total_bytes"],
            current["unique_sources"],
            current["unique_destinations"],
            current["unique_destination_ports"],
            current["node_count"],
            current["edge_count"],
        ]

        next_features = [
            next_state["sessions"],
            next_state["total_packets"],
            next_state["total_bytes"],
            next_state["unique_sources"],
            next_state["unique_destinations"],
            next_state["unique_destination_ports"],
            next_state["node_count"],
            next_state["edge_count"],
        ]

        X.append(current_features)
        y.append(next_features)

    return X, y