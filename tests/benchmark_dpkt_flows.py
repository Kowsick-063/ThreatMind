import time

from preprocessing.ingestion.dpkt_temporal_flow_builder import (
    build_temporal_flows,
)


PCAP_FILE = (
    "data/raw/cicids2017/pcap/"
    "Friday-WorkingHours.pcap"
)

MAX_PACKETS = 1_000_000
WINDOW_SECONDS = 5


def main():
    start_time = time.perf_counter()

    windows_processed = 0
    flows_processed = 0
    packets_processed = 0

    for window_start, flows in build_temporal_flows(
        PCAP_FILE,
        window_seconds=WINDOW_SECONDS,
        max_packets=MAX_PACKETS,
    ):
        windows_processed += 1
        flows_processed += len(flows)

        for flow in flows:
            packets_processed += flow["packets"]

    elapsed = time.perf_counter() - start_time

    print()
    print("========== DPKT FLOW BENCHMARK ==========")
    print(f"Packet limit:     {MAX_PACKETS:,}")
    print(f"Packets in flows: {packets_processed:,}")
    print(f"Windows:          {windows_processed:,}")
    print(f"Flows:            {flows_processed:,}")
    print(f"Time:             {elapsed:.2f} seconds")

    if elapsed > 0:
        print(
            f"Speed:            "
            f"{MAX_PACKETS / elapsed:,.0f} packets/sec"
        )

    print("==========================================")

    
if __name__ == "__main__":
    main()