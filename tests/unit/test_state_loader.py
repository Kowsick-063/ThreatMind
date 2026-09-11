from preprocessing.state_loader import save_states


states = [
    {
        "window_index": 0,
        "start_time": 100.0,
        "end_time": 105.0,
        "sessions": 2,
        "total_packets": 15,
        "total_bytes": 1500,
        "unique_sources": 2,
        "unique_destinations": 1,
        "unique_destination_ports": 2,
    },
    {
        "window_index": 1,
        "start_time": 105.0,
        "end_time": 110.0,
        "sessions": 1,
        "total_packets": 20,
        "total_bytes": 2000,
        "unique_sources": 1,
        "unique_destinations": 1,
        "unique_destination_ports": 1,
    },
]


count = save_states(
    states,
    window_size=5,
)

print(f"States saved: {count}")