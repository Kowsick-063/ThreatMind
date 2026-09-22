from statistics import mean


# TCP flag bit values
TCP_FIN = 1
TCP_SYN = 2
TCP_RST = 4
TCP_PSH = 8
TCP_ACK = 16


def calculate_network_state(
    window_start: float,
    flows: list[dict],
    window_seconds: int = 5,
) -> dict:
    """
    Convert flows from one temporal window
    into a numerical network state.
    """

    if not flows:

        return {
            "timestamp": window_start,
            "window_size_seconds": window_seconds,

            "flow_count": 0,
            "packet_count": 0,
            "byte_count": 0,

            "unique_sources": 0,
            "unique_destinations": 0,
            "unique_ports": 0,

            "syn_count": 0,
            "ack_count": 0,
            "rst_count": 0,
            "fin_count": 0,
            "psh_count": 0,

            "mean_flow_duration": 0.0,
            "mean_packets_per_flow": 0.0,
            "mean_bytes_per_flow": 0.0,

            "packets_per_second": 0.0,
            "bytes_per_second": 0.0,
        }

    flow_count = len(flows)

    packet_count = sum(
        flow["packets"]
        for flow in flows
    )

    byte_count = sum(
        flow["bytes"]
        for flow in flows
    )

    unique_sources = {
        flow["source_ip"]
        for flow in flows
    }

    unique_destinations = {
        flow["destination_ip"]
        for flow in flows
    }

    unique_ports = set()

    for flow in flows:

        if flow["source_port"] is not None:

            unique_ports.add(
                flow["source_port"]
            )

        if flow["destination_port"] is not None:

            unique_ports.add(
                flow["destination_port"]
            )

    syn_count = 0
    ack_count = 0
    rst_count = 0
    fin_count = 0
    psh_count = 0

    for flow in flows:

        for flags in flow["tcp_flags"]:

            if flags & TCP_SYN:
                syn_count += 1

            if flags & TCP_ACK:
                ack_count += 1

            if flags & TCP_RST:
                rst_count += 1

            if flags & TCP_FIN:
                fin_count += 1

            if flags & TCP_PSH:
                psh_count += 1

    durations = [
        flow["duration"]
        for flow in flows
    ]

    packets_per_flow = [
        flow["packets"]
        for flow in flows
    ]

    bytes_per_flow = [
        flow["bytes"]
        for flow in flows
    ]

    return {
        "timestamp": window_start,
        "window_size_seconds": window_seconds,

        "flow_count": flow_count,
        "packet_count": packet_count,
        "byte_count": byte_count,

        "unique_sources": len(
            unique_sources
        ),

        "unique_destinations": len(
            unique_destinations
        ),

        "unique_ports": len(
            unique_ports
        ),

        "syn_count": syn_count,
        "ack_count": ack_count,
        "rst_count": rst_count,
        "fin_count": fin_count,
        "psh_count": psh_count,

        "mean_flow_duration": mean(
            durations
        ),

        "mean_packets_per_flow": mean(
            packets_per_flow
        ),

        "mean_bytes_per_flow": mean(
            bytes_per_flow
        ),

        "packets_per_second": (
            packet_count
            / window_seconds
        ),

        "bytes_per_second": (
            byte_count
            / window_seconds
        ),
    }