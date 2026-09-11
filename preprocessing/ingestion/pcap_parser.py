from pathlib import Path

from scapy.all import IP, TCP, UDP, rdpcap


def parse_pcap(file_path: str) -> list[dict]:
    """
    Parse a PCAP file and extract basic network packet information.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PCAP file not found: {file_path}")

    packets = rdpcap(str(path))

    records = []

    for packet in packets:

        if IP not in packet:
            continue

        record = {
            "timestamp": float(packet.time),
            "source_ip": packet[IP].src,
            "destination_ip": packet[IP].dst,
            "protocol": packet[IP].proto,
            "source_port": None,
            "destination_port": None,
            "packet_length": len(packet),
            "tcp_flags": None,
        }

        if TCP in packet:
            record["source_port"] = packet[TCP].sport
            record["destination_port"] = packet[TCP].dport
            record["tcp_flags"] = str(packet[TCP].flags)

        elif UDP in packet:
            record["source_port"] = packet[UDP].sport
            record["destination_port"] = packet[UDP].dport

        records.append(record)

    return records