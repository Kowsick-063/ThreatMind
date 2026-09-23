import csv
import time
from pathlib import Path

from preprocessing.ingestion.dpkt_temporal_flow_builder import (
    build_temporal_flows,
)
from preprocessing.temporal.network_state_builder import (
    calculate_network_state,
)


PCAP_FILE = (
    "data/raw/cicids2017/pcap/"
    "Friday-WorkingHours.pcap"
)

OUTPUT_FILE = (
    "data/processed/"
    "threatmind_friday_temporal_states.csv"
)

WINDOW_SECONDS = 5


FIELDNAMES = [
    "timestamp",
    "window_size_seconds",
    "flow_count",
    "packet_count",
    "byte_count",
    "unique_sources",
    "unique_destinations",
    "unique_ports",
    "syn_count",
    "ack_count",
    "rst_count",
    "fin_count",
    "psh_count",
    "mean_flow_duration",
    "mean_packets_per_flow",
    "mean_bytes_per_flow",
    "packets_per_second",
    "bytes_per_second",
    "mean_ttl",
    "std_ttl",
    "mean_tcp_window",
    "std_tcp_window",
    "mean_payload_size",
    "std_payload_size",
    "fragment_count",
    "mean_iat",
    "std_iat",
    "max_iat",
]


def main():
    output_path = Path(OUTPUT_FILE)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_time = time.perf_counter()

    state_count = 0
    total_flows = 0
    total_packets = 0
    total_bytes = 0

    print("==========================================")
    print("ThreatMind Temporal State Builder")
    print("==========================================")
    print(f"Input:  {PCAP_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Window: {WINDOW_SECONDS} seconds")
    print()
    print("Starting full PCAP processing...")
    print("This may take several minutes.")
    print()

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()

        for window_start, flows in build_temporal_flows(
            PCAP_FILE,
            window_seconds=WINDOW_SECONDS,
        ):

            state = calculate_network_state(
                window_start,
                flows,
                window_seconds=WINDOW_SECONDS,
            )

            writer.writerow(state)

            state_count += 1
            total_flows += len(flows)
            total_packets += state["packet_count"]
            total_bytes += state["byte_count"]

            if state_count % 1000 == 0:

                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                print(
                    f"States: {state_count:,} | "
                    f"Flows: {total_flows:,} | "
                    f"Packets: {total_packets:,} | "
                    f"Time: {elapsed:.1f}s"
                )

    elapsed = time.perf_counter() - start_time

    print()
    print("==========================================")
    print("PROCESSING COMPLETE")
    print("==========================================")
    print(f"States:        {state_count:,}")
    print(f"Flows:         {total_flows:,}")
    print(f"Packets:       {total_packets:,}")
    print(f"Bytes:         {total_bytes:,}")
    print(f"Time:          {elapsed:.2f} seconds")
    print(f"Output:        {output_path}")
    print("==========================================")


if __name__ == "__main__":
    main()