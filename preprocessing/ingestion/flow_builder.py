from collections import defaultdict
from pathlib import Path

from preprocessing.ingestion.pcap_stream import stream_pcap


def create_flow_key(packet):
    """
    Create a bidirectional flow key.

    A -> B and B -> A belong to the same flow.
    """

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


def build_flows(pcap_file: str, max_packets: int | None = None):
    """
    Build flows from a PCAP stream.

    max_packets is used for testing so we don't process
    the entire 8.32 GB PCAP.
    """

    flows = {}

    packet_count = 0

    for packet in stream_pcap(pcap_file):

        packet_count += 1

        if max_packets is not None and packet_count > max_packets:
            break

        key = create_flow_key(packet)

        if key not in flows:
            flows[key] = {
                "source_ip": packet["source_ip"],
                "destination_ip": packet["destination_ip"],
                "source_port": packet["source_port"],
                "destination_port": packet["destination_port"],
                "protocol": packet["protocol"],
                "start_time": packet["timestamp"],
                "end_time": packet["timestamp"],
                "packets": 0,
                "bytes": 0,
                "tcp_flags": set(),
            }

        flow = flows[key]

        flow["packets"] += 1
        flow["bytes"] += packet["packet_length"]

        flow["start_time"] = min(
            flow["start_time"],
            packet["timestamp"]
        )

        flow["end_time"] = max(
            flow["end_time"],
            packet["timestamp"]
        )

        if packet["tcp_flags"]:
            flow["tcp_flags"].add(packet["tcp_flags"])

    # Finalize flows
    finalized_flows = []

    for flow in flows.values():

        flow["duration"] = (
            flow["end_time"] - flow["start_time"]
        )

        flow["tcp_flags"] = ",".join(
            sorted(flow["tcp_flags"])
        )

        finalized_flows.append(flow)

    return finalized_flows