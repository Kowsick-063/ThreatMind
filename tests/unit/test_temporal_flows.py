from preprocessing.ingestion.temporal_flow_builder import (
    build_temporal_flows,
)


PCAP_FILE = (
    "data/raw/cicids2017/pcap/"
    "Friday-WorkingHours.pcap"
)


def main():

    total_windows = 0
    total_flows = 0

    for window_start, flows in build_temporal_flows(
        PCAP_FILE,
        window_seconds=5,
        max_packets=10000,
    ):

        total_windows += 1
        total_flows += len(flows)

        print(
            f"Window: {window_start:.3f} | "
            f"Flows: {len(flows)}"
        )

        if total_windows >= 10:
            break

    print()
    print(f"Windows processed: {total_windows}")
    print(f"Flows processed: {total_flows}")


if __name__ == "__main__":
    main()
    