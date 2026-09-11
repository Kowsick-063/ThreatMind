from preprocessing.temporal.window_generator import generate_time_windows


sessions = [
    {
        "source_ip": "192.168.1.10",
        "destination_ip": "192.168.1.20",
        "source_port": 5000,
        "destination_port": 80,
        "protocol": "6",
        "start_time": 100.0,
        "end_time": 102.0,
        "duration": 2.0,
        "packets": 10,
        "bytes_transferred": 1000,
    },
    {
        "source_ip": "192.168.1.11",
        "destination_ip": "192.168.1.20",
        "source_port": 5001,
        "destination_port": 443,
        "protocol": "6",
        "start_time": 103.0,
        "end_time": 104.0,
        "duration": 1.0,
        "packets": 5,
        "bytes_transferred": 500,
    },
    {
        "source_ip": "192.168.1.12",
        "destination_ip": "192.168.1.30",
        "source_port": 5002,
        "destination_port": 22,
        "protocol": "6",
        "start_time": 107.0,
        "end_time": 109.0,
        "duration": 2.0,
        "packets": 20,
        "bytes_transferred": 2000,
    },
]


states = generate_time_windows(
    sessions,
    window_size=5,
)

print("Temporal states:", len(states))

for state in states:
    print(state)