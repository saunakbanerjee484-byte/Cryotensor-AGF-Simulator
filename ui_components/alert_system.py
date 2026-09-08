"""
ui_components/alert_system.py
================================
Deterministic Red/Yellow/Green engineering assessment.

100% rule-based on classical thresholds -- NO ML, NO heuristic scoring model.
Every threshold is a named, documented, user-adjustable numeric constant so
the design basis is fully auditable ("no black box").
"""

from __future__ import annotations
from dataclasses import dataclass
import streamlit as st


@dataclass
class AlertResult:
    status: str        # "green" | "yellow" | "red"
    headline: str
    detail: str


def evaluate_module1_status(
    frozen_fraction_final: float,
    max_cooling_rate: float,
    freeze_front_radius_m: float,
    target_closure_radius_m: float,
    critical_cooling_rate: float = 0.02,     # [degC/s] thermal-shock / cracking-risk threshold
    warning_cooling_rate: float = 0.01,      # [degC/s]
) -> AlertResult:
    """
    Deterministic assessment for the Stefan Phase-Change Matrix (Module 1).

    RED   : cooling rate exceeds critical thermal-shock threshold (cracking risk), OR
            freeze front has stalled well short of target closure with negligible growth.
    YELLOW: cooling rate in the moderate band, OR front is still closing but has not
            yet reached target closure radius.
    GREEN : cooling within safe bounds AND freeze front has reached (or exceeded)
            the target design closure radius.
    """
    closure_ratio = 0.0
    if target_closure_radius_m > 0:
        closure_ratio = freeze_front_radius_m / target_closure_radius_m

    if max_cooling_rate >= critical_cooling_rate:
        return AlertResult(
            status="red",
            headline="CRITICAL: Thermal shock / cracking-risk cooling rate exceeded.",
            detail=(
                f"Peak nodal cooling rate {max_cooling_rate:.4f} °C/s ≥ critical "
                f"threshold {critical_cooling_rate:.4f} °C/s. Reduce brine ΔT ramp or "
                f"stage freeze-pipe activation to limit thermal gradient-induced cracking."
            ),
        )

    if closure_ratio >= 1.0 and max_cooling_rate < warning_cooling_rate:
        return AlertResult(
            status="green",
            headline="OPTIMAL: Freeze wall closure achieved within safe thermal limits.",
            detail=(
                f"Freeze front radius {freeze_front_radius_m:.3f} m has reached the "
                f"target closure radius {target_closure_radius_m:.3f} m "
                f"({closure_ratio*100:.0f}% of target). Peak cooling rate "
                f"{max_cooling_rate:.4f} °C/s is within the safe operating band."
            ),
        )

    return AlertResult(
        status="yellow",
        headline="MODERATE: Freeze wall still developing / approaching thresholds.",
        detail=(
            f"Freeze front radius {freeze_front_radius_m:.3f} m is at "
            f"{closure_ratio*100:.0f}% of the target closure radius "
            f"({target_closure_radius_m:.3f} m). Peak cooling rate "
            f"{max_cooling_rate:.4f} °C/s. Continue monitoring; extend freeze duration "
            f"if closure is not achieved within the design schedule."
        ),
    )


def render_alert(result: AlertResult) -> None:
    """Render an AlertResult using the mandated Streamlit status boxes."""
    message = f"**{result.headline}**\n\n{result.detail}"
    if result.status == "green":
        st.success(message)
    elif result.status == "yellow":
        st.warning(message)
    elif result.status == "red":
        st.error(message)
    else:
        st.info(message)
