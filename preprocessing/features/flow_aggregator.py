from collections import defaultdict


def create_flow_key(packet: dict) -> tuple:
    """
    Create a network flow identifier using the standard 5-tuple.
    """

    return (
        packet["source_ip"],
        packet["destination_ip"],
        packet["source_port"],
        packet["destination_port"],
        packet["protocol"],
    )


def aggregate_flows(packets: list[dict]) -> list[dict]:
    """
    Aggregate individual packets into network flows.
    """

    flows = defaultdict(list)

    for packet in packets:
        key = create_flow_key(packet)
        flows[key].append(packet)

    sessions = []

    for flow_key, flow_packets in flows.items():

        first_packet = min(
            flow_packets,
            key=lambda packet: packet["timestamp"],
        )

        last_packet = max(
            flow_packets,
            key=lambda packet: packet["timestamp"],
        )

        total_packets = len(flow_packets)

        total_bytes = sum(
            packet["packet_length"]
            for packet in flow_packets
        )

        duration = (
            last_packet["timestamp"]
            - first_packet["timestamp"]
        )

        sessions.append(
            {
                "source_ip": flow_key[0],
                "destination_ip": flow_key[1],
                "source_port": flow_key[2],
                "destination_port": flow_key[3],
                "protocol": flow_key[4],
                "start_time": first_packet["timestamp"],
                "end_time": last_packet["timestamp"],
                "duration": duration,
                "packets": total_packets,
                "bytes_transferred": total_bytes,
            }
        )

    return sessions