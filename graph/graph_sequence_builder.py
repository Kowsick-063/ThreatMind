from __future__ import annotations

import json
from pathlib import Path

from graph.graph_builder import (
    FRIDAY_PCAP,
    LABELED_STATES,
    build_dynamic_graph_from_pcap,
    save_graph_snapshots,
)
from graph.graph_snapshot import GraphSequence, GraphSnapshot


GRAPH_DIRECTORY = Path("data/processed/graphs")
SNAPSHOT_FILE = GRAPH_DIRECTORY / "friday_graph_snapshots.jsonl"
SEQUENCE_FILE = GRAPH_DIRECTORY / "friday_graph_sequences.jsonl"


def build_graph_sequences(
    snapshots: list[GraphSnapshot],
    sequence_length: int = 60,
    window_seconds: int = 5,
) -> list[GraphSequence]:
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive.")
    ordered = sorted(snapshots, key=lambda snapshot: snapshot.timestamp)
    sequences = []
    block: list[GraphSnapshot] = []

    def flush() -> None:
        if len(block) < sequence_length:
            return
        for end in range(sequence_length, len(block) + 1):
            sequences.append(GraphSequence(block[end - sequence_length:end]))

    for snapshot in ordered:
        if block and snapshot.timestamp - block[-1].timestamp != window_seconds:
            flush()
            block = []
        block.append(snapshot)
    flush()
    return sequences


def save_graph_sequences(
    sequences: list[GraphSequence],
    output_file: str | Path,
) -> None:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for sequence in sequences:
            file.write(json.dumps(sequence.to_dict(), sort_keys=True) + "\n")


def build_graph_sequences_from_pcap(
    pcap_file: str | Path = FRIDAY_PCAP,
    labeled_states_file: str | Path = LABELED_STATES,
    sequence_length: int = 60,
    window_seconds: int = 5,
) -> tuple[list[GraphSnapshot], list[GraphSequence]]:
    snapshots = build_dynamic_graph_from_pcap(
        pcap_file,
        window_seconds,
        labeled_states_file,
    )
    sequences = build_graph_sequences(
        snapshots,
        sequence_length,
        window_seconds,
    )
    save_graph_snapshots(snapshots, SNAPSHOT_FILE)
    save_graph_sequences(sequences, SEQUENCE_FILE)
    return snapshots, sequences


def main() -> None:
    snapshots, sequences = build_graph_sequences_from_pcap()
    print("ThreatMind Dynamic Graph Builder")
    print("=================================")
    print(f"Snapshots: {len(snapshots):,}")
    print(f"Sequences: {len(sequences):,}")
    print(f"Snapshot artifact: {SNAPSHOT_FILE}")
    print(f"Sequence artifact: {SEQUENCE_FILE}")


if __name__ == "__main__":
    main()