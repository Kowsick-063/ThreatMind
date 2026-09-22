import time
from pathlib import Path

import dpkt


PCAP_FILE = Path(
    "data/raw/cicids2017/pcap/Friday-WorkingHours.pcap"
)

MAX_PACKETS = 1_000_000


def main():
    start_time = time.perf_counter()

    packet_count = 0
    ip_packet_count = 0
    tcp_count = 0
    udp_count = 0

    with open(PCAP_FILE, "rb") as file:
        reader = dpkt.pcapng.Reader(file)

        for timestamp, raw_packet in reader:
            packet_count += 1

            try:
                ethernet = dpkt.ethernet.Ethernet(raw_packet)

                if not isinstance(ethernet.data, dpkt.ip.IP):
                    continue

                ip = ethernet.data
                ip_packet_count += 1

                if isinstance(ip.data, dpkt.tcp.TCP):
                    tcp_count += 1

                elif isinstance(ip.data, dpkt.udp.UDP):
                    udp_count += 1

            except (dpkt.dpkt.UnpackError, ValueError):
                continue

            if packet_count >= MAX_PACKETS:
                break

    elapsed = time.perf_counter() - start_time

    print()
    print("========== DPKT PCAP-NG BENCHMARK ==========")
    print(f"Packets processed: {packet_count:,}")
    print(f"IP packets:        {ip_packet_count:,}")
    print(f"TCP packets:       {tcp_count:,}")
    print(f"UDP packets:       {udp_count:,}")
    print(f"Time:              {elapsed:.2f} seconds")

    if elapsed > 0:
        packets_per_second = packet_count / elapsed
        print(
            f"Speed:             "
            f"{packets_per_second:,.0f} packets/sec"
        )

    print("=============================================")


if __name__ == "__main__":
    main() 