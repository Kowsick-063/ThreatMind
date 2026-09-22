from pathlib import Path


PCAP_DIR = Path(
    "data/raw/cicids2017/pcap"
)


def main():
    if not PCAP_DIR.exists():
        raise FileNotFoundError(
            f"PCAP directory not found: {PCAP_DIR}"
        )

    files = []

    for path in PCAP_DIR.rglob("*"):
        if path.is_file():
            suffix = path.suffix.lower()

            if suffix in {
                ".pcap",
                ".pcapng",
                ".cap",
            }:
                files.append(path)

    files.sort()

    print("=" * 80)
    print("CIC-IDS2017 PCAP INVENTORY")
    print("=" * 80)

    print(
        f"Directory: {PCAP_DIR.resolve()}"
    )

    print(
        f"PCAP files found: {len(files)}"
    )

    print()

    total_size = 0

    for index, path in enumerate(
        files,
        start=1,
    ):
        size = path.stat().st_size
        total_size += size

        size_gb = (
            size / (1024 ** 3)
        )

        print(
            f"{index:>3}. "
            f"{path.name}"
        )

        print(
            f"     Size: "
            f"{size:,} bytes "
            f"({size_gb:.2f} GB)"
        )

    print()
    print("=" * 80)

    total_gb = (
        total_size / (1024 ** 3)
    )

    print(
        f"Total PCAP size: "
        f"{total_gb:.2f} GB"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()