import json
from pathlib import Path

import numpy as np

from graph_experiment.dataset import (
    DATASET_FILE,
    GRAPH_FEATURE_NAMES,
    NODE_FEATURE_NAMES,
    EDGE_FEATURE_NAMES,
    LABEL_NAMES,
    METADATA_FILE,
    GraphDataset,
)


def inspect(dataset_file: str | Path = DATASET_FILE, metadata_file: str | Path = METADATA_FILE):
    dataset = GraphDataset(dataset_file)
    metadata = json.loads(Path(metadata_file).read_text(encoding="utf-8"))
    report = {
        "number_of_sequences": len(dataset),
        "sequence_length": metadata["sequence_length"],
        "node_feature_dimension": len(NODE_FEATURE_NAMES),
        "edge_feature_dimension": len(EDGE_FEATURE_NAMES),
        "graph_feature_dimension": len(GRAPH_FEATURE_NAMES),
        "label_distribution": {},
        "split_sizes": {},
        "node_count": {},
        "edge_count": {},
        "missing_or_invalid_values": {},
    }
    for split in ("train", "validation", "test"):
        values = dataset.split(split)
        sequence_indices = values["sequence_indices"]
        labels = dataset.archive["snapshot__labels"][sequence_indices].reshape(-1)
        report["split_sizes"][split] = int(sequence_indices.shape[0])
        report["label_distribution"][split] = {
            label: int(np.sum(labels == index))
            for index, label in enumerate(LABEL_NAMES)
        }
        node_offsets = dataset.archive["snapshot__node_offsets"]
        edge_offsets = dataset.archive["snapshot__edge_offsets"]
        selected = sequence_indices.reshape(-1)
        node_counts = np.diff(node_offsets)[selected]
        edge_counts = np.diff(edge_offsets)[selected]
        report["node_count"][split] = {
            "min": int(node_counts.min()),
            "max": int(node_counts.max()),
            "average": float(node_counts.mean()),
        }
        report["edge_count"][split] = {
            "min": int(edge_counts.min()),
            "max": int(edge_counts.max()),
            "average": float(edge_counts.mean()),
        }
        invalid = {}
        for key in ("node_features", "edge_features", "graph_features"):
            source_key = f"snapshot__{key}"
            invalid[key] = int(
                np.size(dataset.archive[source_key])
                - np.isfinite(dataset.archive[source_key]).sum()
            )
        invalid["timestamps"] = int(
            np.size(dataset.archive["snapshot__timestamps"])
            - np.isfinite(dataset.archive["snapshot__timestamps"]).sum()
        )
        report["missing_or_invalid_values"][split] = invalid
    return report


if __name__ == "__main__":
    print(json.dumps(inspect(), indent=2))