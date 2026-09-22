from datetime import datetime, timezone

import dpkt


PCAP_FILE = (
    "data/raw/cicids2017/pcap/"
    "Friday-WorkingHours.pcap"
)


def main():
    print("=" * 70)
    print("FRIDAY PCAP TIME RANGE")
    print("=" * 70)

    first_timestamp = None
    last_timestamp = None
    packet_count = 0

    with open(PCAP_FILE, "rb") as file:
        reader = dpkt.pcapng.Reader(file)

        for timestamp, _ in reader:
            packet_count += 1

            if first_timestamp is None:
                first_timestamp = float(timestamp)

            last_timestamp = float(timestamp)

    print()
    print(f"Packets: {packet_count:,}")
    print()

    print(
        "First packet Unix timestamp:",
        first_timestamp,
    )

    print(
        "First packet UTC:",
        datetime.fromtimestamp(
            first_timestamp,
            tz=timezone.utc,
        ),
    )

    print()

    print(
        "Last packet Unix timestamp:",
        last_timestamp,
    )

    print(
        "Last packet UTC:",
        datetime.fromtimestamp(
            last_timestamp,
            tz=timezone.utc,
        ),
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()