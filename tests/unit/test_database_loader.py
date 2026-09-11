from preprocessing.database_loader import save_sessions


sessions = [
    {
        "source_ip": "192.168.1.10",
        "destination_ip": "192.168.1.20",
        "source_port": 5000,
        "destination_port": 80,
        "protocol": 6,
        "start_time": 1.0,
        "end_time": 3.0,
        "duration": 2.0,
        "packets": 3,
        "bytes_transferred": 450,
    }
]


count = save_sessions(sessions)

print(f"Sessions saved: {count}")