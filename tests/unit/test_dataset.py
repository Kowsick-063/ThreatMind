from forecasting.baselines.dataset import build_transition_dataset


states = [
    {
        "sessions": 2,
        "total_packets": 15,
        "total_bytes": 1500,
        "unique_sources": 2,
        "unique_destinations": 1,
        "unique_destination_ports": 2,
        "node_count": 3,
        "edge_count": 2,
    },
    {
        "sessions": 1,
        "total_packets": 20,
        "total_bytes": 2000,
        "unique_sources": 1,
        "unique_destinations": 1,
        "unique_destination_ports": 1,
        "node_count": 2,
        "edge_count": 1,
    },
    {
        "sessions": 3,
        "total_packets": 30,
        "total_bytes": 3000,
        "unique_sources": 2,
        "unique_destinations": 2,
        "unique_destination_ports": 3,
        "node_count": 4,
        "edge_count": 3,
    },
]


X, y = build_transition_dataset(states)

print("Training samples:", len(X))
print("Features per sample:", len(X[0]))
print("First X:", X[0])
print("First y:", y[0])