from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

import numpy as np


SNAPSHOT_FILE = Path("data/processed/graphs/friday_graph_snapshots.jsonl")
SEQUENCE_FILE = Path("data/processed/graphs/friday_graph_sequences.jsonl")
DATASET_FILE = Path("data/processed/graphs/graph_sequence_dataset.npz")
METADATA_FILE = Path("data/processed/graphs/graph_sequence_dataset_metadata.json")

NODE_FEATURE_NAMES = [
    "packets_sent",
    "packets_received",
    "bytes_sent",
    "bytes_received",
    "flows_in",
    "flows_out",
    "degree",
    "in_degree",
    "out_degree",
    "syn_count",
    "ack_count",
    "rst_count",
    "fin_count",
    "psh_count",
    "unique_destinations",
    "unique_sources",
    "unique_ports",
]
EDGE_FEATURE_NAMES = [
    "protocol",
    "source_port",
    "destination_port",
    "packets",
    "bytes",
    "flow_count",
    "syn_count",
    "ack_count",
    "rst_count",
    "fin_count",
    "psh_count",
]
GRAPH_FEATURE_NAMES = [
    "node_count",
    "edge_count",
    "unique_sources",
    "unique_destinations",
    "total_packets",
    "total_bytes",
    "average_degree",
    "max_degree",
    "average_in_degree",
    "average_out_degree",
]
LABEL_NAMES = ["BENIGN", "Bot", "PortScan", "DDoS"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABEL_NAMES)}
SEQUENCE_LENGTH = 60
WINDOW_SECONDS = 5


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}."
                ) from error


def _numeric(value: Any) -> float:
    if isinstance(value, list):
        return float(np.mean([float(item) for item in value])) if value else 0.0
    if value is None:
        return 0.0
    return float(value)


def _snapshot_arrays(snapshot: dict[str, Any]):
    nodes = sorted(snapshot.get("nodes", []), key=lambda node: str(node["ip"]))
    node_ids = [str(node["ip"]) for node in nodes]
    node_index = {ip: index for index, ip in enumerate(node_ids)}
    node_values = np.asarray(
        [[_numeric(node.get(name, 0.0)) for name in NODE_FEATURE_NAMES] for node in nodes],
        dtype=np.float32,
    )

    edges = sorted(
        snapshot.get("edges", []),
        key=lambda edge: (str(edge["source_ip"]), str(edge["destination_ip"])),
    )
    edge_index = []
    edge_values = []
    for edge in edges:
        source = str(edge["source_ip"])
        destination = str(edge["destination_ip"])
        if source not in node_index or destination not in node_index:
            raise ValueError("Edge endpoint is absent from the snapshot node list.")
        edge_index.append([node_index[source], node_index[destination]])
        edge_values.append([_numeric(edge.get(name, 0.0)) for name in EDGE_FEATURE_NAMES])

    return (
        node_ids,
        node_values,
        np.asarray(edge_index, dtype=np.int64).reshape(-1, 2),
        np.asarray(edge_values, dtype=np.float32).reshape(-1, len(EDGE_FEATURE_NAMES)),
    )


def _snapshot_boundaries() -> tuple[float, float, list[float]]:
    timestamps = [
        float(snapshot["timestamp"])
        for snapshot in _iter_jsonl(SNAPSHOT_FILE)
    ]
    if not timestamps:
        raise ValueError("No graph snapshots were found.")
    if any(right <= left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError("Graph snapshots are not strictly chronological.")
    train_index = int(round(len(timestamps) * 0.70))
    validation_index = int(round(len(timestamps) * 0.85))
    return timestamps[train_index], timestamps[validation_index], timestamps


def _split_for_sequence(
    start_timestamp: float,
    end_timestamp: float,
    train_boundary: float,
    validation_boundary: float,
) -> str | None:
    if end_timestamp < train_boundary:
        return "train"
    if start_timestamp >= train_boundary and end_timestamp < validation_boundary:
        return "validation"
    if start_timestamp >= validation_boundary:
        return "test"
    return None


def _empty_split() -> dict[str, list[Any]]:
    return {
        "node_features": [],
        "node_ids": [],
        "node_offsets": [],
        "edge_index": [],
        "edge_features": [],
        "edge_offsets": [],
        "graph_features": [],
        "labels": [],
        "is_attack": [],
        "timestamps": [],
        "start_timestamps": [],
        "end_timestamps": [],
    }


def _append_sequence(target: dict[str, list[Any]], sequence: dict[str, Any]) -> None:
    node_offset = [0]
    edge_offset = [0]
    nodes = []
    node_ids = []
    edges = []
    edge_features = []
    graph_features = []
    labels = []
    is_attack = []
    timestamps = []

    for snapshot in sequence["snapshots"]:
        ids, node_values, edge_index, edge_values = _snapshot_arrays(snapshot)
        nodes.append(node_values)
        node_ids.extend(ids)
        edges.append(edge_index)
        edge_features.append(edge_values)
        node_offset.append(node_offset[-1] + len(ids))
        edge_offset.append(edge_offset[-1] + len(edge_index))
        graph_features.append(
            [_numeric(snapshot["graph_features"].get(name, 0.0)) for name in GRAPH_FEATURE_NAMES]
        )
        label = snapshot.get("attack_label")
        if label not in LABEL_TO_ID:
            raise ValueError(f"Unknown graph label: {label!r}")
        labels.append(LABEL_TO_ID[label])
        is_attack.append(int(snapshot.get("is_attack", label != "BENIGN")))
        timestamps.append(float(snapshot["timestamp"]))

    target["node_features"].append(np.concatenate(nodes, axis=0))
    target["node_ids"].append(node_ids)
    target["node_offsets"].append(node_offset)
    target["edge_index"].append(
        np.concatenate(edges, axis=0) if edges else np.empty((0, 2), dtype=np.int64)
    )
    target["edge_features"].append(np.concatenate(edge_features, axis=0))
    target["edge_offsets"].append(edge_offset)
    target["graph_features"].append(np.asarray(graph_features, dtype=np.float32))
    target["labels"].append(labels)
    target["is_attack"].append(is_attack)
    target["timestamps"].append(timestamps)
    target["start_timestamps"].append(float(sequence["start_timestamp"]))
    target["end_timestamps"].append(float(sequence["end_timestamp"]))


def _finalize_split(split: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    sequence_count = len(split["labels"])
    return {
        "node_features": np.concatenate(split["node_features"], axis=0).astype(np.float32),
        "node_ids": np.asarray([ip for ids in split["node_ids"] for ip in ids]),
        "node_offsets": np.asarray(split["node_offsets"], dtype=np.int64),
        "edge_index": np.concatenate(split["edge_index"], axis=0).astype(np.int64),
        "edge_features": np.concatenate(split["edge_features"], axis=0).astype(np.float32),
        "edge_offsets": np.asarray(split["edge_offsets"], dtype=np.int64),
        "graph_features": np.concatenate(split["graph_features"], axis=0).astype(np.float32),
        "labels": np.asarray(split["labels"], dtype=np.int64).reshape(sequence_count, SEQUENCE_LENGTH),
        "is_attack": np.asarray(split["is_attack"], dtype=np.int64).reshape(sequence_count, SEQUENCE_LENGTH),
        "timestamps": np.asarray(split["timestamps"], dtype=np.float64).reshape(sequence_count, SEQUENCE_LENGTH),
        "start_timestamps": np.asarray(split["start_timestamps"], dtype=np.float64),
        "end_timestamps": np.asarray(split["end_timestamps"], dtype=np.float64),
    }


class GraphDataset:
    def __init__(self, dataset_file: str | Path = DATASET_FILE):
        self.dataset_file = Path(dataset_file)
        self.archive = np.load(self.dataset_file, allow_pickle=False)
        self.arrays = {key: self.archive[key] for key in self.archive.files}

    def split(self, name: str) -> dict[str, np.ndarray]:
        return {
            key[len(name) + 2:]: self.arrays[key]
            for key in self.arrays
            if key.startswith(f"{name}__")
        }

    def __len__(self) -> int:
        return int(self.arrays["sequence_count"].item())

    def snapshot(self, index: int) -> dict[str, np.ndarray | float | int]:
        node_start, node_end = self.arrays["snapshot__node_offsets"][index:index + 2]
        edge_start, edge_end = self.arrays["snapshot__edge_offsets"][index:index + 2]
        return {
            "node_features": self.arrays["snapshot__node_features"][node_start:node_end],
            "edge_index": self.arrays["snapshot__edge_index"][edge_start:edge_end],
            "edge_features": self.arrays["snapshot__edge_features"][edge_start:edge_end],
            "graph_features": self.arrays["snapshot__graph_features"][index],
            "label": int(self.arrays["snapshot__labels"][index]),
            "is_attack": int(self.arrays["snapshot__is_attack"][index]),
            "timestamp": float(self.arrays["snapshot__timestamps"][index]),
        }

    def sequence(self, split: str, index: int) -> list[dict[str, np.ndarray | float | int]]:
        indices = self.arrays[f"{split}__sequence_indices"][index]
        return [self.snapshot(int(snapshot_index)) for snapshot_index in indices]

    def target(self, split: str, index: int) -> dict[str, np.ndarray | float | int]:
        indices = self.arrays[f"{split}__sequence_indices"][index]
        return self.snapshot(int(indices[-1]) + 1)


def prepare_graph_dataset(
    sequence_file: str | Path = SEQUENCE_FILE,
    snapshot_file: str | Path = SNAPSHOT_FILE,
    output_file: str | Path = DATASET_FILE,
    metadata_file: str | Path = METADATA_FILE,
) -> dict[str, Any]:
    global SEQUENCE_FILE, SNAPSHOT_FILE
    SEQUENCE_FILE = Path(sequence_file)
    SNAPSHOT_FILE = Path(snapshot_file)
    snapshots = list(_iter_jsonl(SNAPSHOT_FILE))
    if not snapshots:
        raise ValueError("No graph snapshots were found.")
    timestamps = np.asarray(
        [float(snapshot["timestamp"]) for snapshot in snapshots],
        dtype=np.float64,
    )
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Graph snapshots are not strictly chronological.")
    train_boundary_index = int(round(len(snapshots) * 0.70))
    validation_boundary_index = int(round(len(snapshots) * 0.85))

    node_values = []
    node_ids = []
    node_offsets = [0]
    edge_indices = []
    edge_values = []
    edge_offsets = [0]
    graph_values = []
    labels = []
    is_attack = []
    valid_snapshot_indices = []
    invalid = 0
    for snapshot_index, snapshot in enumerate(snapshots):
        try:
            ids, values, edge_index, values_edge = _snapshot_arrays(snapshot)
            label = snapshot.get("attack_label")
            if label not in LABEL_TO_ID:
                raise ValueError(f"Unknown graph label: {label!r}")
            graph_values.append(
                [_numeric(snapshot["graph_features"].get(name, 0.0)) for name in GRAPH_FEATURE_NAMES]
            )
            node_values.append(values)
            node_ids.extend(ids)
            node_offsets.append(node_offsets[-1] + len(ids))
            edge_indices.append(edge_index)
            edge_values.append(values_edge)
            edge_offsets.append(edge_offsets[-1] + len(edge_index))
            labels.append(LABEL_TO_ID[label])
            is_attack.append(int(snapshot.get("is_attack", label != "BENIGN")))
            valid_snapshot_indices.append(snapshot_index)
        except (KeyError, TypeError, ValueError):
            invalid += 1

    if len(valid_snapshot_indices) != len(snapshots):
        raise ValueError("Invalid snapshots cannot be omitted from chronological indexing.")

    sequence_indices = {"train": [], "validation": [], "test": []}
    discarded_cross_boundary = 0
    block_start = 0
    boundaries = list(np.flatnonzero(np.diff(timestamps) != WINDOW_SECONDS) + 1)
    block_ends = boundaries + [len(snapshots)]
    for block_end in block_ends:
        for start in range(block_start, block_end - SEQUENCE_LENGTH + 1):
            end = start + SEQUENCE_LENGTH
            if end < train_boundary_index:
                split = "train"
            elif start >= train_boundary_index and end < validation_boundary_index:
                split = "validation"
            elif start >= validation_boundary_index and end < len(snapshots):
                split = "test"
            else:
                discarded_cross_boundary += 1
                continue
            sequence_indices[split].append(np.arange(start, end, dtype=np.int64))
        block_start = block_end

    finalized = {
        name: np.asarray(values, dtype=np.int64).reshape(-1, SEQUENCE_LENGTH)
        for name, values in sequence_indices.items()
    }
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "sequence_count": np.asarray(sum(len(values) for values in finalized.values()), dtype=np.int64),
        "snapshot__node_features": np.concatenate(node_values, axis=0).astype(np.float32),
        "snapshot__node_ids": np.asarray(node_ids),
        "snapshot__node_offsets": np.asarray(node_offsets, dtype=np.int64),
        "snapshot__edge_index": np.concatenate(edge_indices, axis=0).astype(np.int64),
        "snapshot__edge_features": np.concatenate(edge_values, axis=0).astype(np.float32),
        "snapshot__edge_offsets": np.asarray(edge_offsets, dtype=np.int64),
        "snapshot__graph_features": np.asarray(graph_values, dtype=np.float32),
        "snapshot__labels": np.asarray(labels, dtype=np.int64),
        "snapshot__is_attack": np.asarray(is_attack, dtype=np.int64),
        "snapshot__timestamps": timestamps,
    }
    for name, values in finalized.items():
        arrays[f"{name}__sequence_indices"] = values
    np.savez_compressed(output_path, **arrays)

    metadata = {
        "source_sequence_file": str(sequence_file),
        "source_snapshot_file": str(snapshot_file),
        "output_file": str(output_file),
        "sequence_length": SEQUENCE_LENGTH,
        "window_seconds": WINDOW_SECONDS,
        "node_feature_names": NODE_FEATURE_NAMES,
        "edge_feature_names": EDGE_FEATURE_NAMES,
        "graph_feature_names": GRAPH_FEATURE_NAMES,
        "label_names": LABEL_NAMES,
        "label_encoding": LABEL_TO_ID,
        "representation": {
            "snapshots": "each graph snapshot stored once in chronological order",
            "sequence_indices": "fixed [sequence, 60] indices into snapshot arrays",
            "node_features": "flattened snapshot arrays with node_offsets[snapshot+1]",
            "node_ids": "flattened in the same order as snapshot node_features",
            "edge_index": "directed local [source_node_index, destination_node_index] pairs per snapshot",
            "edge_features": "flattened snapshot arrays with edge_offsets[snapshot+1]",
            "variable_graph_sizes": "ragged flattened snapshot arrays; no padding",
            "multi_value_edge_attributes": "numeric mean of sorted values; empty values become 0",
        },
        "preprocessing": {
            "normalization": "none",
            "scaler_fit": "none",
            "feature_order_preserved": True,
        },
        "chronological_split": {
            "method": "snapshot boundaries with non-overlapping complete sequences",
            "train_boundary_timestamp": float(timestamps[train_boundary_index]),
            "validation_boundary_timestamp": float(timestamps[validation_boundary_index]),
            "train_sequences": len(finalized["train"]),
            "validation_sequences": len(finalized["validation"]),
            "test_sequences": len(finalized["test"]),
            "discarded_cross_boundary": discarded_cross_boundary,
        },
        "source_snapshot_count": len(snapshots),
        "invalid_sequences": invalid,
    }
    Path(metadata_file).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata