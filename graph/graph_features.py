from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import networkx as nx


TCP_FIN = 1
TCP_SYN = 2
TCP_RST = 4
TCP_PSH = 8
TCP_ACK = 16

NODE_FEATURE_DEFAULTS = {
    "packets_sent": 0,
    "packets_received": 0,
    "bytes_sent": 0,
    "bytes_received": 0,
    "flows_in": 0,
    "flows_out": 0,
    "degree": 0,
    "in_degree": 0,
    "out_degree": 0,
    "syn_count": 0,
    "ack_count": 0,
    "rst_count": 0,
    "fin_count": 0,
    "psh_count": 0,
    "unique_destinations": 0,
    "unique_sources": 0,
    "unique_ports": 0,
}


def _flag_counts(flags: Iterable[int]) -> dict[str, int]:
    values = list(flags)
    return {
        "syn_count": sum(bool(value & TCP_SYN) for value in values),
        "ack_count": sum(bool(value & TCP_ACK) for value in values),
        "rst_count": sum(bool(value & TCP_RST) for value in values),
        "fin_count": sum(bool(value & TCP_FIN) for value in values),
        "psh_count": sum(bool(value & TCP_PSH) for value in values),
    }


def _flow_flags(flow: dict[str, Any]) -> list[int]:
    flags = flow.get("tcp_flags", set())
    if flags is None:
        return []
    if isinstance(flags, (int, float)):
        return [int(flags)]
    if isinstance(flags, str):
        return [int(value) for value in flags.split(",") if value]
    return [int(value) for value in flags]


def node_features_from_flows(flows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    destinations: dict[str, set[str]] = defaultdict(set)
    sources: dict[str, set[str]] = defaultdict(set)
    ports: dict[str, set[int]] = defaultdict(set)

    def ensure(ip: str) -> dict[str, Any]:
        if ip not in nodes:
            nodes[ip] = {"ip": ip, **NODE_FEATURE_DEFAULTS}
        return nodes[ip]

    for flow in flows:
        source = str(flow["source_ip"])
        destination = str(flow["destination_ip"])
        packets = int(flow.get("packets", 0) or 0)
        byte_count = int(flow.get("bytes", 0) or 0)
        flags = _flag_counts(_flow_flags(flow))
        source_node = ensure(source)
        destination_node = ensure(destination)

        source_node["packets_sent"] += packets
        source_node["bytes_sent"] += byte_count
        source_node["flows_out"] += 1
        destination_node["packets_received"] += packets
        destination_node["bytes_received"] += byte_count
        destination_node["flows_in"] += 1

        for key, value in flags.items():
            source_node[key] += value
            destination_node[key] += value

        destinations[source].add(destination)
        sources[destination].add(source)
        for key in ("source_port", "destination_port"):
            if flow.get(key) is not None:
                ports[source].add(int(flow[key]))
                ports[destination].add(int(flow[key]))

    return {
        ip: {
            **attributes,
            "unique_destinations": len(destinations[ip]),
            "unique_sources": len(sources[ip]),
            "unique_ports": len(ports[ip]),
        }
        for ip, attributes in sorted(nodes.items())
    }


def edge_features_from_flows(flows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    for flow in flows:
        source = str(flow["source_ip"])
        destination = str(flow["destination_ip"])
        key = (source, destination)
        if key not in edges:
            edges[key] = {
                "source_ip": source,
                "destination_ip": destination,
                "protocols": set(),
                "source_ports": set(),
                "destination_ports": set(),
                "packets": 0,
                "bytes": 0,
                "flow_count": 0,
                "syn_count": 0,
                "ack_count": 0,
                "rst_count": 0,
                "fin_count": 0,
                "psh_count": 0,
            }
        edge = edges[key]
        edge["protocols"].add(flow.get("protocol"))
        if flow.get("source_port") is not None:
            edge["source_ports"].add(int(flow["source_port"]))
        if flow.get("destination_port") is not None:
            edge["destination_ports"].add(int(flow["destination_port"]))
        edge["packets"] += int(flow.get("packets", 0) or 0)
        edge["bytes"] += int(flow.get("bytes", 0) or 0)
        edge["flow_count"] += 1
        for key_name, value in _flag_counts(_flow_flags(flow)).items():
            edge[key_name] += value

    def scalar_or_list(values: set[Any]) -> Any:
        values = sorted(value for value in values if value is not None)
        return values[0] if len(values) == 1 else values

    result = {}
    for key, attributes in sorted(edges.items()):
        protocols = scalar_or_list(attributes.pop("protocols"))
        source_ports = scalar_or_list(attributes.pop("source_ports"))
        destination_ports = scalar_or_list(
            attributes.pop("destination_ports")
        )
        result[key] = {
            **attributes,
            "protocol": protocols,
            "source_port": source_ports,
            "destination_port": destination_ports,
        }
    return result


def add_graph_features(graph: nx.DiGraph) -> dict[str, float]:
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    in_degrees = [degree for _, degree in graph.in_degree()]
    out_degrees = [degree for _, degree in graph.out_degree()]
    degrees = [degree for _, degree in graph.degree()]
    total_packets = sum(
        int(attributes.get("packets", 0))
        for _, _, attributes in graph.edges(data=True)
    )
    total_bytes = sum(
        int(attributes.get("bytes", 0))
        for _, _, attributes in graph.edges(data=True)
    )

    def average(values: list[int]) -> float:
        return float(sum(values) / len(values)) if values else 0.0

    return {
        "node_count": float(node_count),
        "edge_count": float(edge_count),
        "unique_sources": float(sum(degree > 0 for degree in out_degrees)),
        "unique_destinations": float(sum(degree > 0 for degree in in_degrees)),
        "total_packets": float(total_packets),
        "total_bytes": float(total_bytes),
        "average_degree": average(degrees),
        "max_degree": float(max(degrees)) if degrees else 0.0,
        "average_in_degree": average(in_degrees),
        "average_out_degree": average(out_degrees),
    }


def attach_node_degrees(graph: nx.DiGraph) -> None:
    for node in graph.nodes:
        graph.nodes[node]["in_degree"] = int(graph.in_degree(node))
        graph.nodes[node]["out_degree"] = int(graph.out_degree(node))
        graph.nodes[node]["degree"] = int(graph.degree(node))