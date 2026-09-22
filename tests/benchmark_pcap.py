import time

from preprocessing.ingestion.temporal_flow_builder import (
    build_temporal_flows,
)


PCAP_FILE = (
    "data/raw/cicids2017/pcap/"
    "Friday-WorkingHours.pcap"
)

MAX_PACKETS = 1_000_000


def main():

    start_time = time.perf_counter()

    packets_processed = 0
    windows_processed = 0
    flows_processed = 0

    for window_start, flows in build_temporal_flows(
        PCAP_FILE,
        window_seconds=5,
        max_packets=MAX_PACKETS,
    ):

        windows_processed += 1
        flows_processed += len(flows)

    elapsed = time.perf_counter() - start_time

    print()
    print("========== PCAP BENCHMARK ==========")
    print(f"Packets limit:     {MAX_PACKETS:,}")
    print(f"Windows:           {windows_processed:,}")
    print(f"Flows:             {flows_processed:,}")
    print(f"Time:              {elapsed:.2f} seconds")

    if elapsed > 0:
        packets_per_second = (
            MAX_PACKETS / elapsed
        )

        print(
            f"Speed:             "
            f"{packets_per_second:,.0f} packets/sec"
        )

    print("=====================================")


if __name__ == "__main__":
    main()