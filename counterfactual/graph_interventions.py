from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from counterfactual.interventions import InterventionSpec, build_intervention


@dataclass
class GraphSnapshotData:
    """In-memory representation of a graph snapshot."""
    timestamp: float
    is_attack: int
    label: int
    node_ids: list[str]
    node_features: np.ndarray  # (N, 17)
    edge_index: np.ndarray     # (E, 2)
    edge_features: np.ndarray  # (E, 11)
    graph_features: np.ndarray # (10,)

    def copy(self) -> GraphSnapshotData:
        return GraphSnapshotData(
            timestamp=float(self.timestamp),
            is_attack=int(self.is_attack),
            label=int(self.label),
            node_ids=list(self.node_ids),
            node_features=self.node_features.copy(),
            edge_index=self.edge_index.copy(),
            edge_features=self.edge_features.copy(),
            graph_features=self.graph_features.copy(),
        )


def apply_graph_intervention(
    snapshot: GraphSnapshotData,
    intervention: InterventionSpec | dict[str, Any] | str,
) -> tuple[GraphSnapshotData, list[int]]:
    """
    Apply a graph intervention to a snapshot copy.
    Returns (new_snapshot, removed_edge_indices).
    Does NOT mutate the original snapshot.
    """
    if not isinstance(intervention, InterventionSpec):
        if isinstance(intervention, dict):
            intervention = build_intervention(intervention.get("type", "NO_ACTION"), intervention.get("target", 0))
        else:
            intervention = build_intervention(str(intervention))

    int_type = intervention.type.upper()
    target = intervention.target

    new_snap = snapshot.copy()
    num_edges = len(new_snap.edge_index)
    if num_edges == 0 or int_type == "NO_ACTION":
        return new_snap, []

    src_indices = new_snap.edge_index[:, 0]
    dst_indices = new_snap.edge_index[:, 1]
    node_ids = new_snap.node_ids

    src_ips = [node_ids[idx] for idx in src_indices]
    dst_ips = [node_ids[idx] for idx in dst_indices]

    # EDGE_FEATURE_NAMES = ["protocol", "source_port", "destination_port", "packets", "bytes", ...]
    # destination_port is index 2
    # source_port is index 1
    dst_ports = new_snap.edge_features[:, 2].astype(np.int64) if new_snap.edge_features.shape[1] > 2 else np.zeros(num_edges, dtype=np.int64)

    keep_mask = np.ones(num_edges, dtype=bool)

    target_str = str(target).strip()

    if int_type == "BLOCK_SOURCE":
        for i, sip in enumerate(src_ips):
            if sip == target_str:
                keep_mask[i] = False

    elif int_type == "BLOCK_DESTINATION":
        for i, dip in enumerate(dst_ips):
            if dip == target_str:
                keep_mask[i] = False

    elif int_type == "BLOCK_PORT":
        try:
            port_val = int(target_str)
            keep_mask = (dst_ports != port_val)
        except ValueError:
            pass

    elif int_type == "ISOLATE_HOST":
        for i in range(num_edges):
            if src_ips[i] == target_str or dst_ips[i] == target_str:
                keep_mask[i] = False

    elif int_type == "DISABLE_CONNECTION":
        # target can be "src->dst" or "src,dst" or tuple
        if "->" in target_str:
            tsrc, tdst = [p.strip() for p in target_str.split("->", 1)]
        elif "," in target_str:
            tsrc, tdst = [p.strip() for p in target_str.split(",", 1)]
        else:
            tsrc, tdst = target_str, ""
        for i in range(num_edges):
            if tdst:
                if src_ips[i] == tsrc and dst_ips[i] == tdst:
                    keep_mask[i] = False
            else:
                if src_ips[i] == tsrc or dst_ips[i] == tsrc:
                    keep_mask[i] = False

    elif int_type == "NETWORK_SEGMENTATION":
        # Restrict cross-subnet communication if target specifies subnets,
        # or isolates cross-boundary communication between internal and external IPs.
        # Target format: e.g. "192.168.10" or "segment_b" or default subnet prefix.
        seg = target_str if target_str and target_str != "0" else "192.168.10."
        for i in range(num_edges):
            sip_in = src_ips[i].startswith(seg)
            dip_in = dst_ips[i].startswith(seg)
            # Cut cross-segment edges (from inside to outside or outside to inside)
            if sip_in != dip_in:
                keep_mask[i] = False

    removed_indices = [i for i, keep in enumerate(keep_mask) if not keep]
    if not removed_indices:
        return new_snap, []

    new_snap.edge_index = new_snap.edge_index[keep_mask]
    new_snap.edge_features = new_snap.edge_features[keep_mask]

    # Recompute graph features
    active_node_indices = np.unique(new_snap.edge_index.reshape(-1)) if len(new_snap.edge_index) > 0 else np.array([], dtype=np.int64)
    active_srcs = set(new_snap.edge_index[:, 0]) if len(new_snap.edge_index) > 0 else set()
    active_dsts = set(new_snap.edge_index[:, 1]) if len(new_snap.edge_index) > 0 else set()

    total_packets = float(np.sum(new_snap.edge_features[:, 3])) if len(new_snap.edge_features) > 0 else 0.0
    total_bytes = float(np.sum(new_snap.edge_features[:, 4])) if len(new_snap.edge_features) > 0 else 0.0
    node_count = float(len(active_node_indices))
    edge_count = float(len(new_snap.edge_index))
    avg_degree = (edge_count / node_count) if node_count > 0 else 0.0

    new_snap.graph_features = np.array([
        node_count,
        edge_count,
        float(len(active_srcs)),
        float(len(active_dsts)),
        total_packets,
        total_bytes,
        avg_degree,
        new_snap.graph_features[7],
        avg_degree / 2.0,
        avg_degree / 2.0,
    ], dtype=np.float32)

    return new_snap, removed_indices
