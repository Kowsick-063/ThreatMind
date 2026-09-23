import csv
from pathlib import Path

from preprocessing.ingestion.temporal_flow_builder import (
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


def main():

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
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

    state_count = 0
    flow_count = 0

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
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
            flow_count += len(flows)

            if state_count % 1000 == 0:
                print(
                    f"States: {state_count:,} | "
                    f"Flows: {flow_count:,}"
                )

    print()
    print("Processing complete.")
    print(f"States: {state_count:,}")
    print(f"Flows: {flow_count:,}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()