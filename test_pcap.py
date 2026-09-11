from preprocessing.ingestion.pcap_parser import parse_pcap

records = parse_pcap("data/raw/sample.pcap")

print(f"Packets extracted: {len(records)}")

for record in records[:5]:
    print(record)