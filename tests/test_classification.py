"""Tests for swing classification rules.

Verifies the deterministic logic — given specific feature inputs,
the classifier returns the expected verdict and findings.
"""

from __future__ import annotations

from tennis_coach.analysis.classification import (
    SwingClassification,
    classify_swing,
)
from tennis_coach.analysis.features import SwingFeatures
from tennis_coach.analysis.types import Measurement, Reliability


def _m(
    value: float | None, reliability: Reliability = Reliability.HIGH, unit: str = ""
) -> Measurement:
    """Test helper for constructing Measurement objects concisely."""
    return Measurement(value=value, reliability=reliability, unit=unit)


def _build_features(
    elbow: Measurement | None = None,
    contact_height: Measurement | None = None,
    hip_shoulder: Measurement | None = None,
    knee: Measurement | None = None,
    head: Measurement | None = None,
    duration: Measurement | None = None,
) -> SwingFeatures:
    """Build a SwingFeatures, defaulting any unset to a healthy reliable value."""
    return SwingFeatures(
        elbow_angle_at_contact=elbow or _m(145.0, unit="°"),
        contact_height_vs_shoulder=contact_height or _m(-0.02),
        hip_shoulder_separation=hip_shoulder or _m(30.0, unit="°"),
        knee_bend_at_contact=knee or _m(155.0, unit="°"),
        head_stability=head or _m(0.01),
        swing_duration_ms=duration or _m(900.0, unit=" ms"),
    )


def test_healthy_swing_is_classified_healthy() -> None:
    """All reliable features within range → HEALTHY, no findings."""
    result = classify_swing(_build_features())
    assert result.classification is SwingClassification.HEALTHY
    assert result.findings == []
    assert result.reliable_feature_count == 6


def test_outlier_feature_produces_finding() -> None:
    """A reliable feature outside its range → ISSUES with a finding."""
    features = _build_features(knee=_m(178.0, unit="°"))  # nearly locked
    result = classify_swing(features)
    assert result.classification is SwingClassification.ISSUES
    assert len(result.findings) == 1
    assert result.findings[0].feature == "knee_bend_at_contact"
    assert "above" in result.findings[0].observation


def test_unreliable_measurements_are_ignored() -> None:
    """Outliers on UNRELIABLE measurements should NOT trigger ISSUES."""
    features = _build_features(
        elbow=_m(17.0, reliability=Reliability.UNRELIABLE, unit="°"),  # crazy value, but unreliable
    )
    result = classify_swing(features)
    assert result.classification is SwingClassification.HEALTHY
    # Elbow was unreliable, so reliable_feature_count drops by 1.
    assert result.reliable_feature_count == 5


def test_too_few_reliable_measurements_is_insufficient_data() -> None:
    """Below MIN_RELIABLE_MEASUREMENTS → INSUFFICIENT_DATA verdict."""
    features = _build_features(
        elbow=_m(145.0, reliability=Reliability.UNRELIABLE, unit="°"),
        contact_height=_m(-0.02, reliability=Reliability.UNRELIABLE),
        hip_shoulder=_m(30.0, reliability=Reliability.UNRELIABLE, unit="°"),
        knee=_m(155.0, reliability=Reliability.UNRELIABLE, unit="°"),
        head=_m(0.01, reliability=Reliability.UNRELIABLE),
        duration=_m(900.0, reliability=Reliability.UNRELIABLE, unit=" ms"),
    )
    result = classify_swing(features)
    assert result.classification is SwingClassification.INSUFFICIENT_DATA
    assert result.findings == []


def test_severity_scales_with_distance_from_range() -> None:
    """Values far outside the range get higher severity than borderline ones."""
    # Healthy elbow range is 120-165. Healthy band width = 45.
    # 170 is 5 above → relative_distance ≈ 0.11 → minor
    minor = classify_swing(_build_features(elbow=_m(170.0, unit="°"))).findings[0]
    # 200 is 35 above → relative_distance ≈ 0.78 → major
    major = classify_swing(_build_features(elbow=_m(200.0, unit="°"))).findings[0]
    assert minor.severity == "minor"
    assert major.severity == "major"
