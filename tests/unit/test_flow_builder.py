from preprocessing.ingestion.flow_builder import build_flows


PCAP_FILE = (
    "data/raw/cicids2017/pcap/"
    "Friday-WorkingHours.pcap"
)


def main():

    flows = build_flows(
        PCAP_FILE,
        max_packets=10000
    )

    print(f"Flows created: {len(flows)}")

    print("\nFirst 5 flows:\n")

    for flow in flows[:5]:
        print(flow)


if __name__ == "__main__":
    main()