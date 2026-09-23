from __future__ import annotations

from typing import Any
import numpy as np

from counterfactual.graph_interventions import GraphSnapshotData


def compute_attack_surface(snapshot: GraphSnapshotData) -> dict[str, float]:
    """
    Measurable attack surface properties based on network topology:
    - active_edges: directed communication channels
    - unique_destination_ports: exposed services
    - external_facing_pairs: communication pairs bridging external and internal IPs
    - high_degree_targets: nodes receiving connections from multiple sources
    - composite_attack_surface: normalized combination of exposed channels and ports
    """
    num_edges = len(snapshot.edge_index)
    if num_edges == 0:
        return {
            "exposed_channels": 0.0,
            "unique_services": 0.0,
            "external_bridge_channels": 0.0,
            "attack_surface_score": 0.0,
        }

    dst_ports = snapshot.edge_features[:, 2].astype(np.int64) if snapshot.edge_features.shape[1] > 2 else np.array([])
    unique_ports = len(set(dst_ports.tolist())) if len(dst_ports) > 0 else 0

    node_ids = snapshot.node_ids
    src_indices = snapshot.edge_index[:, 0]
    dst_indices = snapshot.edge_index[:, 1]

    # Count cross-boundary connections (between internal 192.168.10.* and external hosts)
    cross_boundary = 0
    for s_idx, d_idx in zip(src_indices, dst_indices):
        s_ip = node_ids[s_idx]
        d_ip = node_ids[d_idx]
        s_internal = s_ip.startswith("192.168.10.")
        d_internal = d_ip.startswith("192.168.10.")
        if s_internal != d_internal:
            cross_boundary += 1

    # Attack surface score: 0.5 * exposed_channels + 0.3 * unique_services + 0.2 * external_bridge_channels
    attack_surface_score = float(0.5 * num_edges + 0.3 * unique_ports + 0.2 * cross_boundary)

    return {
        "exposed_channels": float(num_edges),
        "unique_services": float(unique_ports),
        "external_bridge_channels": float(cross_boundary),
        "attack_surface_score": round(attack_surface_score, 4),
    }


def compute_communication_paths(
    snapshot: GraphSnapshotData,
    max_hops: int = 2,
    max_paths: int = 200,
) -> list[list[str]]:
    """
    Find directed communication paths in the graph snapshot (1-hop and 2-hop).
    Paths are labeled as graph communication paths.
    """
    num_edges = len(snapshot.edge_index)
    if num_edges == 0:
        return []

    node_ids = snapshot.node_ids
    src_indices = snapshot.edge_index[:, 0]
    dst_indices = snapshot.edge_index[:, 1]

    adjacency: dict[int, list[int]] = {}
    for s_idx, d_idx in zip(src_indices, dst_indices):
        adjacency.setdefault(int(s_idx), []).append(int(d_idx))

    paths: list[list[str]] = []

    # 1-hop paths
    for s_idx, d_idx in zip(src_indices, dst_indices):
        paths.append([node_ids[s_idx], node_ids[d_idx]])
        if len(paths) >= max_paths:
            return paths

    # 2-hop paths (s -> m -> d)
    if max_hops >= 2:
        for s_idx in adjacency:
            for m_idx in adjacency[s_idx]:
                if m_idx in adjacency:
                    for d_idx in adjacency[m_idx]:
                        if d_idx != s_idx:  # avoid cycles
                            paths.append([node_ids[s_idx], node_ids[m_idx], node_ids[d_idx]])
                            if len(paths) >= max_paths:
                                return paths

    return paths


def calculate_graph_impact(
    baseline: GraphSnapshotData,
    counterfactual: GraphSnapshotData,
    removed_edge_indices: list[int],
) -> dict[str, Any]:
    """
    Compute structured graph impact metrics comparing baseline to counterfactual.
    """
    base_nodes = set(baseline.edge_index.reshape(-1)) if len(baseline.edge_index) > 0 else set()
    cf_nodes = set(counterfactual.edge_index.reshape(-1)) if len(counterfactual.edge_index) > 0 else set()
    nodes_removed = max(0, len(base_nodes) - len(cf_nodes))

    edges_removed = len(removed_edge_indices)

    # Volume removed
    if edges_removed > 0 and len(baseline.edge_features) > 0:
        pkts_removed = float(np.sum(baseline.edge_features[removed_edge_indices, 3]))
        bytes_removed = float(np.sum(baseline.edge_features[removed_edge_indices, 4]))
    else:
        pkts_removed = 0.0
        bytes_removed = 0.0

    # Affected endpoints
    affected_sources = set()
    affected_destinations = set()
    affected_ports = set()
    if edges_removed > 0:
        node_ids = baseline.node_ids
        for idx in removed_edge_indices:
            s_idx = baseline.edge_index[idx, 0]
            d_idx = baseline.edge_index[idx, 1]
            affected_sources.add(node_ids[s_idx])
            affected_destinations.add(node_ids[d_idx])
            if baseline.edge_features.shape[1] > 2:
                affected_ports.add(int(baseline.edge_features[idx, 2]))

    # Connectivity change
    base_edges = len(baseline.edge_index)
    cf_edges = len(counterfactual.edge_index)
    connectivity_change = round(float((cf_edges - base_edges) / max(base_edges, 1)), 4)

    # Attack surface
    base_as = compute_attack_surface(baseline)
    cf_as = compute_attack_surface(counterfactual)
    as_reduction = round(base_as["attack_surface_score"] - cf_as["attack_surface_score"], 4)
    as_reduction_pct = round((as_reduction / max(base_as["attack_surface_score"], 1e-6)) * 100.0, 2)

    # Attack paths / communication paths
    base_paths = compute_communication_paths(baseline)
    cf_paths = compute_communication_paths(counterfactual)
    affected_paths = max(0, len(base_paths) - len(cf_paths))
    path_reduction_pct = round((affected_paths / max(len(base_paths), 1)) * 100.0, 2)

    return {
        "nodes_removed": int(nodes_removed),
        "edges_removed": int(edges_removed),
        "packet_volume_removed": round(pkts_removed, 2),
        "byte_volume_removed": round(bytes_removed, 2),
        "affected_sources": sorted(list(affected_sources)),
        "affected_destinations": sorted(list(affected_destinations)),
        "affected_ports": sorted(list(affected_ports)),
        "graph_connectivity_change": connectivity_change,
        "baseline_attack_surface": base_as,
        "counterfactual_attack_surface": cf_as,
        "attack_surface_reduction": as_reduction,
        "attack_surface_reduction_percentage": as_reduction_pct,
        "affected_attack_paths": int(affected_paths),
        "remaining_attack_paths": int(len(cf_paths)),
        "attack_path_reduction": float(affected_paths),
        "attack_path_reduction_percentage": path_reduction_pct,
    }
