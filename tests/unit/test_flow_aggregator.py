from preprocessing.features.flow_aggregator import aggregate_flows


packets = [
    {
        "timestamp": 1.0,
        "source_ip": "192.168.1.10",
        "destination_ip": "192.168.1.20",
        "source_port": 5000,
        "destination_port": 80,
        "protocol": 6,
        "packet_length": 100,
        "tcp_flags": "S",
    },
    {
        "timestamp": 2.0,
        "source_ip": "192.168.1.10",
        "destination_ip": "192.168.1.20",
        "source_port": 5000,
        "destination_port": 80,
        "protocol": 6,
        "packet_length": 200,
        "tcp_flags": "A",
    },
    {
        "timestamp": 3.0,
        "source_ip": "192.168.1.10",
        "destination_ip": "192.168.1.20",
        "source_port": 5000,
        "destination_port": 80,
        "protocol": 6,
        "packet_length": 150,
        "tcp_flags": "PA",
    },
]


sessions = aggregate_flows(packets)

print("Flows created:", len(sessions))
print(sessions[0])