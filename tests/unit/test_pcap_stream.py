from preprocessing.ingestion.pcap_stream import stream_pcap


PCAP_FILE = "data/raw/cicids2017/pcap/Friday-WorkingHours.pcap"


def main():
    packet_count = 0

    for packet in stream_pcap(PCAP_FILE):
        packet_count += 1

        if packet_count <= 5:
            print(packet)

        if packet_count >= 100:
            break

    print(f"\nPackets successfully read: {packet_count}")


if __name__ == "__main__":
    main()