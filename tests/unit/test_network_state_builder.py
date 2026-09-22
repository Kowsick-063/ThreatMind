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


def main():

    states = []

    for window_start, flows in build_temporal_flows(
        PCAP_FILE,
        window_seconds=5,
        max_packets=10000,
    ):

        state = calculate_network_state(
            window_start,
            flows,
        )

        states.append(state)

        if len(states) >= 10:
            break

    for state in states:
        print(state)

    print()
    print(f"States created: {len(states)}")


if __name__ == "__main__":
    main()