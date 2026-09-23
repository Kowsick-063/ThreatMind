from pathlib import Path

import dpkt


def stream_pcap(file_path: str):
    """
    Stream packets from a PCAP-NG file using dpkt.

    Yields normalized packet records without loading
    the complete PCAP into memory.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"PCAP not found: {file_path}"
        )

    with open(path, "rb") as file:
        reader = dpkt.pcapng.Reader(file)
        previous_timestamp = None

        for timestamp, raw_packet in reader:

            try:
                ethernet = dpkt.ethernet.Ethernet(raw_packet)

                # Only process IPv4 packets.
                if not isinstance(
                    ethernet.data,
                    dpkt.ip.IP,
                ):
                    continue

                ip = ethernet.data

                timestamp = float(timestamp)

                if previous_timestamp is None:
                    iat = 0.0
                else:
                    iat = max(
                        0.0,
                        timestamp - previous_timestamp,
                    )

                previous_timestamp = timestamp

                source_ip = socket_address(ip.src)
                destination_ip = socket_address(ip.dst)

                ip_fragmented = int(
                    bool(
                        ip.mf
                        or ip.offset
                    )
                )

                record = {
                    "timestamp": timestamp,
                    "source_ip": source_ip,
                    "destination_ip": destination_ip,
                    "protocol": int(ip.p),
                    "source_port": None,
                    "destination_port": None,
                    "packet_length": len(raw_packet),
                    "tcp_flags": None,
                    "ttl": int(ip.ttl),
                    "tcp_window_size": 0,
                    "payload_size": 0,
                    "ip_fragmented": ip_fragmented,
                    "iat": iat,
                }

                if isinstance(ip.data, dpkt.tcp.TCP):
                    tcp = ip.data

                    record["source_port"] = int(
                        tcp.sport
                    )

                    record["destination_port"] = int(
                        tcp.dport
                    )

                    record["tcp_flags"] = int(
                        tcp.flags
                    )

                    record["tcp_window_size"] = int(
                        tcp.win
                    )

                    record["payload_size"] = len(
                        tcp.data
                    )

                elif isinstance(ip.data, dpkt.udp.UDP):
                    udp = ip.data

                    record["source_port"] = int(
                        udp.sport
                    )

                    record["destination_port"] = int(
                        udp.dport
                    )

                    record["payload_size"] = len(
                        udp.data
                    )

                yield record

            except (
                dpkt.dpkt.UnpackError,
                ValueError,
                IndexError,
            ):
                continue


def socket_address(address: bytes) -> str:
    """
    Convert a 4-byte IPv4 address to dotted notation.
    """

    return ".".join(
        str(byte)
        for byte in address
    )