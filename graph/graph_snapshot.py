from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx


@dataclass
class GraphSnapshot:
    timestamp: float
    window_start: float
    window_end: float
    graph: nx.DiGraph
    graph_features: dict[str, float]
    attack_label: str | None = None
    is_attack: int | None = None

    @property
    def nodes(self) -> dict[str, dict[str, Any]]:
        return {
            str(node): dict(attributes)
            for node, attributes in self.graph.nodes(data=True)
        }

    @property
    def edges(self) -> list[dict[str, Any]]:
        return [
            {
                "source_ip": str(source),
                "destination_ip": str(destination),
                **dict(attributes),
            }
            for source, destination, attributes in self.graph.edges(data=True)
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "attack_label": self.attack_label,
            "is_attack": self.is_attack,
            "nodes": [
                {"ip": ip, **attributes}
                for ip, attributes in sorted(self.nodes.items())
            ],
            "edges": sorted(
                self.edges,
                key=lambda edge: (
                    edge["source_ip"],
                    edge["destination_ip"],
                ),
            ),
            "graph_features": dict(sorted(self.graph_features.items())),
        }


@dataclass
class GraphSequence:
    snapshots: list[GraphSnapshot]

    @property
    def start_timestamp(self) -> float:
        return self.snapshots[0].timestamp

    @property
    def end_timestamp(self) -> float:
        return self.snapshots[-1].timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "sequence_length": len(self.snapshots),
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
        }