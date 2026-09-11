import random


def generate_synthetic_states(
    num_states: int = 1000,
    seed: int = 42,
) -> list[dict]:
    """
    Generate temporally correlated network states
    for development and pipeline testing.

    This dataset is NOT a replacement for real
    cybersecurity datasets.
    """

    random.seed(seed)

    states = []

    sessions = 20
    packets = 200
    bytes_transferred = 20000
    sources = 5
    destinations = 5
    ports = 3

    for index in range(num_states):

        # Small natural changes between consecutive states
        sessions += random.randint(-3, 5)
        packets += random.randint(-20, 40)
        bytes_transferred += random.randint(-2000, 5000)

        sources += random.choice([-1, 0, 0, 1])
        destinations += random.choice([-1, 0, 0, 1])
        ports += random.choice([-1, 0, 0, 1])

        # Keep values realistic
        sessions = max(1, sessions)
        packets = max(10, packets)
        bytes_transferred = max(100, bytes_transferred)
        sources = max(1, sources)
        destinations = max(1, destinations)
        ports = max(1, ports)

        # Approximate graph structure
        node_count = sources + destinations
        edge_count = sessions

        states.append(
            {
                "window_index": index,
                "start_time": index * 5,
                "end_time": (index + 1) * 5,
                "sessions": sessions,
                "total_packets": packets,
                "total_bytes": bytes_transferred,
                "unique_sources": sources,
                "unique_destinations": destinations,
                "unique_destination_ports": ports,
                "node_count": node_count,
                "edge_count": edge_count,
            }
        )

    return states