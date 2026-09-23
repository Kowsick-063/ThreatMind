from preprocessing.ingestion.dpkt_stream import (
    stream_pcap,
)


def create_flow_key(packet: dict):
    """
    Create a bidirectional flow key.

    The two endpoints are sorted so that traffic in
    either direction belongs to the same flow.
    """

    endpoint_a = (
        packet["source_ip"],
        packet["source_port"],
    )

    endpoint_b = (
        packet["destination_ip"],
        packet["destination_port"],
    )

    endpoints = sorted(
        [endpoint_a, endpoint_b]
    )

    return (
        endpoints[0],
        endpoints[1],
        packet["protocol"],
    )


def build_temporal_flows(
    pcap_file: str,
    window_seconds: int = 5,
    max_packets: int | None = None,
):
    """
    Build bidirectional flows inside temporal windows.

    The PCAP is streamed and never loaded completely
    into memory.
    """

    current_window = None

    flows = {}

    packet_count = 0

    for packet in stream_pcap(pcap_file):

        packet_count += 1

        if (
            max_packets is not None
            and packet_count > max_packets
        ):
            break

        timestamp = packet["timestamp"]

        window_start = (
            timestamp // window_seconds
        ) * window_seconds

        if current_window is None:
            current_window = window_start

        if window_start != current_window:

            finalized_flows = []

            for flow in flows.values():

                flow["duration"] = (
                    flow["end_time"]
                    - flow["start_time"]
                )

                finalized_flows.append(flow)

            yield (
                current_window,
                finalized_flows,
            )

            flows = {}

            current_window = window_start

        key = create_flow_key(packet)

        if key not in flows:

            flows[key] = {
                "source_ip": packet[
                    "source_ip"
                ],
                "destination_ip": packet[
                    "destination_ip"
                ],
                "source_port": packet[
                    "source_port"
                ],
                "destination_port": packet[
                    "destination_port"
                ],
                "protocol": packet[
                    "protocol"
                ],
                "start_time": timestamp,
                "end_time": timestamp,
                "packets": 0,
                "bytes": 0,
                "tcp_flags": set(),
                "packet_features": [],
            }

        flow = flows[key]

        flow["packets"] += 1

        flow["bytes"] += packet[
            "packet_length"
        ]

        flow["end_time"] = timestamp

        flow["packet_features"].append(
            {
                "ttl": packet["ttl"],
                "tcp_window_size": packet[
                    "tcp_window_size"
                ],
                "payload_size": packet[
                    "payload_size"
                ],
                "ip_fragmented": packet[
                    "ip_fragmented"
                ],
                "iat": packet["iat"],
            }
        )

        if packet["tcp_flags"] is not None:

            flow["tcp_flags"].add(
                packet["tcp_flags"]
            )

    # Final window
    if flows:

        finalized_flows = []

        for flow in flows.values():

            flow["duration"] = (
                flow["end_time"]
                - flow["start_time"]
            )

            finalized_flows.append(flow)

        yield (
            current_window,
            finalized_flows,
        )