"""Rule-based swing classification.

Pure functions that take SwingFeatures and decide which coaching
strategy applies: healthy form, specific issues, or insufficient data.

The output drives branching in the LangGraph agent, but lives here as
plain functions so it can be tested without any agent infrastructure
and reused by other consumers (e.g. a CLI report tool).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tennis_coach.analysis.features import SwingFeatures
from tennis_coach.analysis.types import Measurement, Reliability


class SwingClassification(str, Enum):
    """Top-level verdict on the swing."""

    HEALTHY = "healthy"
    ISSUES = "issues"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class Finding:
    """A single noted issue with a swing.

    Used both for human-readable coaching and as structured input to
    the LLM (which can phrase it naturally).
    """

    feature: str  # e.g. "knee_bend_at_contact"
    observation: str  # e.g. "knee is nearly locked (174°, target 140-165°)"
    severity: str  # "minor" | "moderate" | "major"


@dataclass(frozen=True)
class ClassificationResult:
    """Output of classify_swing — the verdict plus supporting evidence."""

    classification: SwingClassification
    findings: list[Finding]
    reliable_feature_count: int
    reliability_summary: str  # e.g. "3 of 6 measurements reliable from this camera angle"


# Healthy ranges for reliable measurements. Wide enough to allow
# stylistic variation while flagging clear outliers.
# Values are tuned for amateur-to-intermediate players hitting forehands.
HEALTHY_RANGES: dict[str, tuple[float, float]] = {
    "elbow_angle_at_contact": (120.0, 165.0),  # degrees
    "contact_height_vs_shoulder": (-0.15, 0.10),  # normalized; near-shoulder
    "hip_shoulder_separation": (15.0, 50.0),  # degrees of X-factor
    "knee_bend_at_contact": (140.0, 168.0),  # degrees; some flex required
    "head_stability": (0.0, 0.025),  # normalized Y std-dev
    # swing_duration_ms is informational only — not range-checked.
}

# Minimum reliable measurements required to make a confident judgement.
MIN_RELIABLE_MEASUREMENTS = 2


def classify_swing(features: SwingFeatures) -> ClassificationResult:
    """Classify a swing's form based on its biomechanical measurements.

    Only considers measurements with HIGH or LOW reliability — UNRELIABLE
    measurements are ignored entirely to avoid coaching on noise.

    Args:
        features: SwingFeatures output from extract_features.

    Returns:
        ClassificationResult with verdict and supporting findings.
    """
    # Iterate over the SwingFeatures fields in display order.
    feature_items: list[tuple[str, Measurement]] = [
        ("elbow_angle_at_contact", features.elbow_angle_at_contact),
        ("contact_height_vs_shoulder", features.contact_height_vs_shoulder),
        ("hip_shoulder_separation", features.hip_shoulder_separation),
        ("knee_bend_at_contact", features.knee_bend_at_contact),
        ("head_stability", features.head_stability),
        ("swing_duration_ms", features.swing_duration_ms),
    ]

    reliable = [(name, m) for name, m in feature_items if m.is_usable]
    reliable_count = len(reliable)
    summary = (
        f"{reliable_count} of {len(feature_items)} measurements reliable from this camera angle"
    )

    # Not enough reliable signal to coach with confidence.
    if reliable_count < MIN_RELIABLE_MEASUREMENTS:
        return ClassificationResult(
            classification=SwingClassification.INSUFFICIENT_DATA,
            findings=[],
            reliable_feature_count=reliable_count,
            reliability_summary=summary,
        )

    findings: list[Finding] = []
    for name, m in reliable:
        finding = _check_range(name, m)
        if finding is not None:
            findings.append(finding)

    classification = SwingClassification.ISSUES if findings else SwingClassification.HEALTHY
    return ClassificationResult(
        classification=classification,
        findings=findings,
        reliable_feature_count=reliable_count,
        reliability_summary=summary,
    )


def _check_range(feature_name: str, m: Measurement) -> Finding | None:
    """Return a Finding if the measurement is outside its healthy range."""
    if m.value is None or feature_name not in HEALTHY_RANGES:
        return None

    low, high = HEALTHY_RANGES[feature_name]
    if low <= m.value <= high:
        return None

    # Outside range: characterize how far off, and how serious.
    if m.value < low:
        direction = "below"
        distance = low - m.value
    else:
        direction = "above"
        distance = m.value - high

    range_width = high - low
    relative_distance = distance / range_width if range_width > 0 else 1.0
    severity = (
        "major" if relative_distance > 0.5 else "moderate" if relative_distance > 0.2 else "minor"
    )

    observation = (
        f"{_humanize(feature_name)} is {direction} the healthy range: "
        f"measured {m.value:.1f}{m.unit}, healthy range {low:.0f}-{high:.0f}{m.unit}"
    )
    # Reliability is also part of the evidence for the LLM.
    if m.reliability is Reliability.LOW:
        observation += " (low-reliability measurement; treat with caution)"

    return Finding(feature=feature_name, observation=observation, severity=severity)


def _humanize(feature_name: str) -> str:
    """Convert a snake_case feature name to readable text for findings."""
    mapping = {
        "elbow_angle_at_contact": "Elbow angle at contact",
        "contact_height_vs_shoulder": "Contact height relative to shoulder",
        "hip_shoulder_separation": "Hip-shoulder separation",
        "knee_bend_at_contact": "Knee bend at contact",
        "head_stability": "Head stability",
    }
    return mapping.get(feature_name, feature_name)
