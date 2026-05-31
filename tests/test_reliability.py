"""Tests for the reliability lookup table.

Verifies the contract — every feature must have a mapping for every
camera angle. This protects against forgetting to update the table
when adding new angles.
"""

from __future__ import annotations

import pytest

from tennis_coach.analysis.reliability import (
    ELBOW_ANGLE,
    KNEE_BEND,
    RELIABILITY_TABLE,
    lookup_reliability,
)
from tennis_coach.analysis.types import CameraAngle, Reliability


def test_every_feature_covers_every_angle() -> None:
    """Each feature must have an entry for every CameraAngle value."""
    for feature, angle_map in RELIABILITY_TABLE.items():
        missing = set(CameraAngle) - set(angle_map.keys())
        assert not missing, f"Feature '{feature}' missing angles: {missing}"


def test_elbow_angle_unreliable_from_back() -> None:
    """Sanity check: elbow angle from back view should be marked UNRELIABLE."""
    reliability, note = lookup_reliability(ELBOW_ANGLE, CameraAngle.BACK)
    assert reliability is Reliability.UNRELIABLE
    assert note is not None and "back" in note.lower()


def test_knee_bend_reliable_from_all_angles() -> None:
    """Knee bend works in any view — verify the table reflects that."""
    for angle in CameraAngle:
        reliability, _ = lookup_reliability(KNEE_BEND, angle)
        assert reliability is Reliability.HIGH, f"Knee bend should be HIGH for {angle}"


def test_unknown_feature_raises_keyerror() -> None:
    """Typos in feature names must fail loudly."""
    with pytest.raises(KeyError, match="No reliability mapping"):
        lookup_reliability("nonexistent_feature", CameraAngle.SIDE_LEFT)
