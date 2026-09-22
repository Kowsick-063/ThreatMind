from pathlib import Path

from scapy.all import IP, TCP, UDP, PcapReader


def stream_pcap(file_path: str):
    """
    Stream packets from a PCAP file without loading
    the entire capture into RAM.

    Yields one dictionary per IPv4 packet.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"PCAP not found: {file_path}"
        )

    with PcapReader(str(path)) as packets:

        for packet in packets:

            if IP not in packet:
                continue

            record = {
                "timestamp": float(packet.time),
                "source_ip": packet[IP].src,
                "destination_ip": packet[IP].dst,
                "protocol": int(packet[IP].proto),
                "source_port": None,
                "destination_port": None,
                "packet_length": len(packet),
                "tcp_flags": None,
            }

            if TCP in packet:

                record["source_port"] = int(
                    packet[TCP].sport
                )

                record["destination_port"] = int(
                    packet[TCP].dport
                )

                # Store TCP flags as an integer.
                # This is much faster than converting
                # Scapy Flag objects to strings.
                record["tcp_flags"] = int(
                    packet[TCP].flags
                )

            elif UDP in packet:

                record["source_port"] = int(
                    packet[UDP].sport
                )

                record["destination_port"] = int(
                    packet[UDP].dport
                )

            yield record