from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import pandas as pd

from graph.graph_features import (
    add_graph_features,
    edge_features_from_flows,
    node_features_from_flows,
)
from graph.graph_snapshot import GraphSnapshot
from preprocessing.ingestion.dpkt_temporal_flow_builder import build_temporal_flows


FRIDAY_PCAP = Path("data/raw/cicids2017/pcap/Friday-WorkingHours.pcap")
LABELED_STATES = Path("data/processed/threatmind_friday_labeled_states.csv")


def build_graph_snapshot(
    flows: Iterable[dict[str, Any]],
    window_start: float,
    window_end: float | None = None,
    attack_label: str | None = None,
    is_attack: int | None = None,
) -> GraphSnapshot:
    flow_list = list(flows)
    if window_end is None:
        window_end = window_start + 5.0
    graph = nx.DiGraph()
    for ip, attributes in node_features_from_flows(flow_list).items():
        graph.add_node(ip, **attributes)
    for (source, destination), attributes in edge_features_from_flows(
        flow_list
    ).items():
        graph.add_edge(source, destination, **attributes)

    from graph.graph_features import attach_node_degrees

    attach_node_degrees(graph)
    return GraphSnapshot(
        timestamp=float(window_start),
        window_start=float(window_start),
        window_end=float(window_end),
        graph=graph,
        graph_features=add_graph_features(graph),
        attack_label=attack_label,
        is_attack=is_attack,
    )


def _label_index(path: Path) -> dict[float, tuple[str | None, int | None]]:
    if not path.exists():
        return {}
    data = pd.read_csv(path)
    if "timestamp" not in data.columns:
        return {}
    return {
        float(row.timestamp): (
            str(row.attack_label) if "attack_label" in data.columns else None,
            int(row.is_attack) if "is_attack" in data.columns else None,
        )
        for row in data.itertuples()
    }


def build_dynamic_graph(
    flows: Iterable[dict[str, Any]],
    window_seconds: int = 5,
    labels: dict[float, tuple[str | None, int | None]] | None = None,
) -> list[GraphSnapshot]:
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive.")
    windows: dict[float, list[dict[str, Any]]] = {}
    for flow in flows:
        timestamp = flow.get("timestamp", flow.get("start_time"))
        if timestamp is None:
            raise ValueError("Each flow requires timestamp or start_time.")
        window_start = (float(timestamp) // window_seconds) * window_seconds
        windows.setdefault(window_start, []).append(flow)

    snapshots = []
    for window_start in sorted(windows):
        label, is_attack = (labels or {}).get(window_start, (None, None))
        snapshots.append(
            build_graph_snapshot(
                windows[window_start],
                window_start,
                window_start + window_seconds,
                label,
                is_attack,
            )
        )
    return snapshots


def build_dynamic_graph_from_pcap(
    pcap_file: str | Path = FRIDAY_PCAP,
    window_seconds: int = 5,
    labeled_states_file: str | Path = LABELED_STATES,
) -> list[GraphSnapshot]:
    labels = _label_index(Path(labeled_states_file))
    snapshots = []
    for window_start, flows in build_temporal_flows(
        str(pcap_file),
        window_seconds=window_seconds,
    ):
        label, is_attack = labels.get(float(window_start), (None, None))
        snapshots.append(
            build_graph_snapshot(
                flows,
                window_start,
                window_start + window_seconds,
                label,
                is_attack,
            )
        )
    return snapshots


def save_graph_snapshots(
    snapshots: list[GraphSnapshot],
    output_file: str | Path,
) -> None:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for snapshot in snapshots:
            file.write(json.dumps(snapshot.to_dict(), sort_keys=True) + "\n")