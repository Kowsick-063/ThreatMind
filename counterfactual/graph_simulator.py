from __future__ import annotations

from typing import Any
import numpy as np

from counterfactual.graph_interventions import GraphSnapshotData, apply_graph_intervention
from counterfactual.graph_counterfactual import calculate_graph_impact, compute_attack_surface, compute_communication_paths
from counterfactual.interventions import InterventionSpec, build_intervention
from graph_experiment.dataset import GraphDataset, DATASET_FILE

_dataset_cache: GraphDataset | None = None


def get_graph_dataset() -> GraphDataset:
    global _dataset_cache
    if _dataset_cache is None:
        _dataset_cache = GraphDataset(DATASET_FILE)
    return _dataset_cache


def get_graph_snapshot_by_timestamp(timestamp: float | int | None) -> GraphSnapshotData | None:
    """Retrieve historical graph snapshot by timestamp (exact or closest within tolerance)."""
    dataset = get_graph_dataset()
    timestamps = dataset.arrays["snapshot__timestamps"]

    if timestamp is None:
        idx = len(timestamps) - 1
    else:
        target = float(timestamp)
        diffs = np.abs(timestamps - target)
        idx = int(np.argmin(diffs))
        if diffs[idx] > 30.0:
            return None

    raw_snap = dataset.snapshot(idx)
    node_start, node_end = dataset.arrays["snapshot__node_offsets"][idx:idx + 2]
    node_ids = dataset.arrays["snapshot__node_ids"][node_start:node_end].tolist()

    return GraphSnapshotData(
        timestamp=float(raw_snap["timestamp"]),
        is_attack=int(raw_snap["is_attack"]),
        label=int(raw_snap["label"]),
        node_ids=node_ids,
        node_features=raw_snap["node_features"],
        edge_index=raw_snap["edge_index"],
        edge_features=raw_snap["edge_features"],
        graph_features=raw_snap["graph_features"],
    )


def simulate_graph_counterfactual(
    intervention: InterventionSpec | dict[str, Any] | str,
    timestamp: float | int | None = None,
) -> dict[str, Any]:
    """
    Simulate the topological and attack-surface effect of an intervention
    on the real network graph at the specified timestamp.
    """
    if not isinstance(intervention, InterventionSpec):
        if isinstance(intervention, dict):
            intervention = build_intervention(intervention.get("type", "NO_ACTION"), intervention.get("target", 0))
        else:
            intervention = build_intervention(str(intervention))

    baseline = get_graph_snapshot_by_timestamp(timestamp)
    if baseline is None:
        return {
            "intervention": {
                "type": intervention.type,
                "target": intervention.target,
            },
            "graph_available": False,
            "graph_impact": None,
        }

    cf_snapshot, removed_indices = apply_graph_intervention(baseline, intervention)
    impact = calculate_graph_impact(baseline, cf_snapshot, removed_indices)

    base_summary = {
        "timestamp": baseline.timestamp,
        "is_attack": baseline.is_attack,
        "nodes": len(baseline.node_ids),
        "edges": len(baseline.edge_index),
        "attack_surface_score": impact["baseline_attack_surface"]["attack_surface_score"],
    }
    cf_summary = {
        "timestamp": cf_snapshot.timestamp,
        "is_attack": cf_snapshot.is_attack,
        "nodes": int(baseline.graph_features[0] - impact["nodes_removed"]),
        "edges": len(cf_snapshot.edge_index),
        "attack_surface_score": impact["counterfactual_attack_surface"]["attack_surface_score"],
    }

    return {
        "intervention": {
            "type": intervention.type,
            "target": intervention.target,
            "operational_cost": intervention.operational_cost,
            "service_disruption_cost": intervention.service_disruption_cost,
        },
        "graph_available": True,
        "baseline_graph_summary": base_summary,
        "counterfactual_graph_summary": cf_summary,
        "graph_impact": impact,
    }
