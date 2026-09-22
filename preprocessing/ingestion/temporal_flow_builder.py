from collections import defaultdict
from pathlib import Path

from preprocessing.ingestion.pcap_stream import stream_pcap


def create_flow_key(packet):
    endpoint_a = (
        packet["source_ip"],
        packet["source_port"],
    )

    endpoint_b = (
        packet["destination_ip"],
        packet["destination_port"],
    )

    endpoints = tuple(sorted([endpoint_a, endpoint_b]))

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
    Process PCAP packets chronologically and create
    flows inside fixed timestamp windows.

    Only one time window is kept in memory at a time.
    """

    current_window = None
    flows = {}

    packet_count = 0

    for packet in stream_pcap(pcap_file):

        packet_count += 1

        if max_packets is not None and packet_count > max_packets:
            break

        timestamp = packet["timestamp"]

        window_start = (
            timestamp // window_seconds
        ) * window_seconds

        # First window
        if current_window is None:
            current_window = window_start

        # New time window
        if window_start != current_window:

            yield current_window, list(flows.values())

            flows = {}
            current_window = window_start

        key = create_flow_key(packet)

        if key not in flows:
            flows[key] = {
                "source_ip": packet["source_ip"],
                "destination_ip": packet["destination_ip"],
                "source_port": packet["source_port"],
                "destination_port": packet["destination_port"],
                "protocol": packet["protocol"],
                "start_time": timestamp,
                "end_time": timestamp,
                "packets": 0,
                "bytes": 0,
                "tcp_flags": set(),
            }

        flow = flows[key]

        flow["packets"] += 1
        flow["bytes"] += packet["packet_length"]

        flow["end_time"] = timestamp

        if packet["tcp_flags"]:
            flow["tcp_flags"].add(packet["tcp_flags"])

    # Flush final window
    if flows:
        yield current_window, list(flows.values())