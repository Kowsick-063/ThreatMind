from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from counterfactual.interventions import InterventionSpec


TRAFFIC_REDUCTION_FIELDS = {
    "flow_count",
    "packet_count",
    "total_packet_count",
    "byte_count",
    "total_byte_count",
    "unique_sources",
    "unique_destinations",
    "unique_ports",
    "unique_destination_ports",
    "syn_count",
    "ack_count",
    "rst_count",
    "fin_count",
    "psh_count",
    "syn_rate",
    "ack_rate",
    "rst_rate",
    "fin_rate",
    "psh_rate",
    "mean_flow_packets_per_sec",
    "mean_flow_bytes_per_sec",
    "packets_per_second",
    "bytes_per_second",
    "mean_packets_per_flow",
    "mean_bytes_per_flow",
    "mean_packet_size",
}


def _as_mapping(state: Mapping[str, Any] | dict[str, Any]) -> dict[str, Any]:
    return dict(deepcopy(state))


def _scale_value(value: Any, factor: float) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return max(float(value) * factor, 0.0)
    return float(value) * factor


def _apply_reduction(
    state: dict[str, Any],
    factors: dict[str, float],
    default_factor: float,
) -> dict[str, Any]:
    transformed = _as_mapping(state)
    for key in TRAFFIC_REDUCTION_FIELDS:
        if key in transformed:
            factor = factors.get(key, default_factor)
            transformed[key] = _scale_value(transformed[key], factor)
    return transformed


def apply_intervention(
    state: Mapping[str, Any] | dict[str, Any],
    intervention: InterventionSpec | str,
) -> dict[str, Any]:
    """Return a deterministic, simulation-only transformed state without mutating the source."""

    if isinstance(intervention, str):
        from counterfactual.interventions import build_intervention

        intervention = build_intervention(intervention)

    transformed = _as_mapping(state)
    if intervention.type == "NO_ACTION":
        return transformed

    if intervention.type == "BLOCK_PORT":
        return _apply_reduction(transformed, {"unique_ports": 0.35, "syn_rate": 0.5, "ack_rate": 0.45}, 0.6)

    if intervention.type == "BLOCK_SOURCE":
        return _apply_reduction(transformed, {"unique_sources": 0.4, "flow_count": 0.45, "packet_count": 0.45, "byte_count": 0.45}, 0.55)

    if intervention.type == "BLOCK_DESTINATION":
        return _apply_reduction(transformed, {"unique_destinations": 0.45, "flow_count": 0.5, "packet_count": 0.5, "byte_count": 0.5}, 0.6)

    if intervention.type == "ISOLATE_HOST":
        return _apply_reduction(transformed, {"flow_count": 0.25, "packet_count": 0.25, "byte_count": 0.2, "unique_sources": 0.3, "unique_destinations": 0.3, "unique_ports": 0.3}, 0.35)

    if intervention.type == "DISABLE_CONNECTION":
        return _apply_reduction(transformed, {"flow_count": 0.4, "packet_count": 0.35, "byte_count": 0.3, "unique_sources": 0.35, "unique_destinations": 0.35}, 0.4)

    if intervention.type == "NETWORK_SEGMENTATION":
        return _apply_reduction(transformed, {"flow_count": 0.3, "packet_count": 0.3, "byte_count": 0.28, "unique_sources": 0.45, "unique_destinations": 0.45, "syn_rate": 0.4}, 0.5)

    return transformed


__all__ = ["apply_intervention"]
