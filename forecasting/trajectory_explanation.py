"""Trajectory explanation and narrative synthesis for ThreatMind.

Explicitly distinguishes:
  - Observed evidence (ground-truth state at t=0)
  - Predicted state (multi-step forecast from production LSTM)
  - Predicted attack probability (calibrated ML risk output)
  - MITRE evidence confidence (feature-grounded ATT&CK matching)

Guarantees cautious security terminology ("predicted elevated attack probability",
never "attack will occur").
"""

from __future__ import annotations

from typing import Any


def generate_trajectory_explanation(
    current_state: dict[str, float],
    trajectory: list[dict[str, Any]],
    trajectory_summary: dict[str, Any],
    current_timestamp: float | int | str | None = None,
) -> dict[str, Any]:
    """Generate a structured trajectory explanation distinguishing evidence types.

    Parameters
    ----------
    current_state:
        Observed network state features at t=0.
    trajectory:
        List of 12 fused trajectory points (H1..H12).
    trajectory_summary:
        Trajectory summary statistics dictionary.
    current_timestamp:
        Optional timestamp of initial state.

    Returns
    -------
    dict with summary, observations, trajectory_changes, mitre_evidence, uncertainties.
    """
    ts_label = f"t={current_timestamp}" if current_timestamp is not None else "t=0"

    # ------------------------------------------------------------------
    # 1. Observed evidence (at t=0)
    # ------------------------------------------------------------------
    fc = current_state.get("flow_count", 0.0)
    pc = current_state.get("packet_count", 0.0)
    bc = current_state.get("byte_count", 0.0)
    up = current_state.get("unique_ports", 0.0)
    pps = current_state.get("packets_per_second", 0.0)
    syn = current_state.get("syn_count", 0.0)
    rst = current_state.get("rst_count", 0.0)

    observations = [
        f"Observed network state at {ts_label}: flow_count={fc:.0f}, packet_count={pc:.0f}, "
        f"byte_count={bc:.0f}, unique_ports={up:.0f}.",
        f"Traffic rates: {pps:.1f} packets/sec, TCP flags: SYN={syn:.0f}, RST={rst:.0f}.",
    ]

    # ------------------------------------------------------------------
    # 2. Trajectory changes (predicted state shifts H1..H12)
    # ------------------------------------------------------------------
    trajectory_changes: list[str] = []
    if trajectory:
        h1_state = trajectory[0].get("state_features", {})
        h12_state = trajectory[-1].get("state_features", {})

        fc_h12 = h12_state.get("flow_count", fc)
        up_h12 = h12_state.get("unique_ports", up)
        pps_h12 = h12_state.get("packets_per_second", pps)

        trajectory_changes.append(
            f"Predicted state evolution across H1–H12 (production LSTM): flow_count moves from "
            f"{fc:.0f} to {fc_h12:.0f} (delta: {fc_h12 - fc:+.0f})."
        )
        trajectory_changes.append(
            f"Predicted unique destination ports shift from {up:.0f} to {up_h12:.0f} by H12."
        )
        trajectory_changes.append(
            f"Predicted packets_per_second shifts from {pps:.1f} to {pps_h12:.1f} by H12."
        )

        peak_h = trajectory_summary.get("peak_horizon", 1)
        peak_p = trajectory_summary.get("peak_attack_probability", 0.0)
        trend = trajectory_summary.get("probability_trend", "stable")
        trajectory_changes.append(
            f"Predicted attack probability trend is {trend}, reaching a peak of "
            f"{peak_p:.4f} at horizon H{peak_h} (predicted elevated attack probability)."
        )

    # ------------------------------------------------------------------
    # 3. MITRE evidence (aggregated across horizons)
    # ------------------------------------------------------------------
    technique_map: dict[str, dict[str, Any]] = {}
    for point in trajectory:
        h = point.get("horizon", 1)
        for tech in point.get("mitre_techniques", []):
            tid = tech.get("technique_id")
            if not tid:
                continue
            if tid not in technique_map:
                technique_map[tid] = {
                    "technique_id": tid,
                    "technique_name": tech.get("technique_name", ""),
                    "tactic": tech.get("tactic", ""),
                    "evidence_confidence": tech.get("confidence", 0.0),
                    "horizons_observed": [h],
                    "evidence_samples": tech.get("evidence", [])[:2],
                    "confidence_source": (
                        "Evidence conditions evaluated on predicted network state; "
                        "independent of model attack probability"
                    ),
                }
            else:
                technique_map[tid]["horizons_observed"].append(h)
                technique_map[tid]["evidence_confidence"] = max(
                    technique_map[tid]["evidence_confidence"],
                    tech.get("confidence", 0.0),
                )

    mitre_evidence = list(technique_map.values())

    # ------------------------------------------------------------------
    # 4. Uncertainties (explicit limitations and caveats)
    # ------------------------------------------------------------------
    uncertainties: list[str] = [
        "Autoregressive forecasting uncertainty accumulates across horizons H1 to H12.",
        "Delta-LSTM provides secondary change signals; production LSTM is the primary absolute state world model.",
        "MITRE ATT&CK confidence reflects evidence conditions met in the predicted state, separate from the model's predicted attack probability.",
    ]
    # Collect point-level uncertainties (e.g. category limitations)
    for point in trajectory:
        for u in point.get("uncertainties", []):
            if u not in uncertainties:
                uncertainties.append(u)

    # ------------------------------------------------------------------
    # 5. Summary narrative
    # ------------------------------------------------------------------
    peak_prob = trajectory_summary.get("peak_attack_probability", 0.0)
    peak_hor = trajectory_summary.get("peak_horizon", 1)
    prob_trend = trajectory_summary.get("probability_trend", "stable")
    cat_trans = trajectory_summary.get("category_transition", [])
    cats_str = ", ".join(cat_trans) if cat_trans else "BENIGN"

    summary = (
        f"Observed state at {ts_label} exhibits {fc:.0f} flows and {up:.0f} unique ports. "
        f"Across the 12-horizon forecast window (60 seconds), the production LSTM world model "
        f"projects a {prob_trend} attack probability trend, with a peak predicted elevated "
        f"attack probability of {peak_prob:.4f} at horizon H{peak_hor} (not a confirmed attack). "
        f"Category progression: [{cats_str}]. "
        f"MITRE mapping identified {len(mitre_evidence)} technique(s) grounded in state evidence conditions "
        f"(confidence is strictly decoupled from forecast probability)."
    )

    return {
        "summary": summary,
        "observations": observations,
        "trajectory_changes": trajectory_changes,
        "mitre_evidence": mitre_evidence,
        "uncertainties": uncertainties,
    }


__all__ = ["generate_trajectory_explanation"]
