from pathlib import Path
from scapy.all import IP, TCP, UDP, PcapReader


def stream_pcap(file_path: str):
    """
    Read a PCAP packet-by-packet without loading
    the entire file into memory.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PCAP not found: {file_path}")

    with PcapReader(str(path)) as packets:
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

            yield record