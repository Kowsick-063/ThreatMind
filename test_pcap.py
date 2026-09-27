from pathlib import Path

import pytest

from preprocessing.ingestion.pcap_parser import parse_pcap


PCAP_PATH = Path("data/raw/sample.pcap")


@pytest.mark.skipif(
    not PCAP_PATH.exists(),
    reason="Sample PCAP not available",
)
def test_parse_sample_pcap():
    records = parse_pcap(str(PCAP_PATH))

    assert records is not None
    assert len(records) > 0