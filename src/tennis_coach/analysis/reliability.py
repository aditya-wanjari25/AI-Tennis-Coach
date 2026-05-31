"""Reliability mapping: (feature, camera_angle) → Reliability + note.

This is the single source of truth for what's measurable from each
camera angle. Every feature in `features.py` consults this table to
decide what reliability to attach to its measurement.

Adding a new feature: add an entry to RELIABILITY_TABLE.
Adding a new camera angle: add a column to every existing entry.
"""

from __future__ import annotations

from tennis_coach.analysis.types import CameraAngle, Reliability

# Feature name keys — match the field names in SwingFeatures.
# Defined as constants so a typo in a key is caught by tooling.
ELBOW_ANGLE = "elbow_angle_at_contact"
CONTACT_HEIGHT = "contact_height_vs_shoulder"
HIP_SHOULDER_SEP = "hip_shoulder_separation"
KNEE_BEND = "knee_bend_at_contact"
HEAD_STABILITY = "head_stability"
SWING_DURATION = "swing_duration_ms"

# Notes shown to the coaching LLM when a measurement is low/unreliable.
# Kept short and factual — the LLM will phrase them naturally.
_NOTE_DEPTH_FROM_BACK = (
    "Back-view footage compresses the swing's depth axis; this measurement is degraded."
)
_NOTE_DEPTH_FROM_FRONT = (
    "Front-view footage compresses the swing's depth axis; this measurement is degraded."
)
_NOTE_ROTATION_INVISIBLE = (
    "Body rotation cannot be measured from this angle; the player's torso is not visible edge-on."
)
_NOTE_THREE_QUARTER = (
    "Three-quarter camera angle introduces moderate distortion in this measurement."
)

# (feature, angle) → (Reliability, note_or_None)
RELIABILITY_TABLE: dict[str, dict[CameraAngle, tuple[Reliability, str | None]]] = {
    ELBOW_ANGLE: {
        CameraAngle.SIDE_LEFT: (Reliability.HIGH, None),
        CameraAngle.SIDE_RIGHT: (Reliability.HIGH, None),
        CameraAngle.FRONT: (Reliability.LOW, _NOTE_DEPTH_FROM_FRONT),
        CameraAngle.BACK: (Reliability.UNRELIABLE, _NOTE_DEPTH_FROM_BACK),
        CameraAngle.THREE_QUARTER: (Reliability.LOW, _NOTE_THREE_QUARTER),
        CameraAngle.UNKNOWN: (Reliability.LOW, "Camera angle unknown; treat with caution."),
    },
    CONTACT_HEIGHT: {
        CameraAngle.SIDE_LEFT: (Reliability.HIGH, None),
        CameraAngle.SIDE_RIGHT: (Reliability.HIGH, None),
        CameraAngle.FRONT: (Reliability.LOW, _NOTE_DEPTH_FROM_FRONT),
        CameraAngle.BACK: (Reliability.LOW, _NOTE_DEPTH_FROM_BACK),
        CameraAngle.THREE_QUARTER: (Reliability.LOW, _NOTE_THREE_QUARTER),
        CameraAngle.UNKNOWN: (Reliability.LOW, None),
    },
    HIP_SHOULDER_SEP: {
        CameraAngle.SIDE_LEFT: (Reliability.HIGH, None),
        CameraAngle.SIDE_RIGHT: (Reliability.HIGH, None),
        CameraAngle.FRONT: (Reliability.UNRELIABLE, _NOTE_ROTATION_INVISIBLE),
        CameraAngle.BACK: (Reliability.UNRELIABLE, _NOTE_ROTATION_INVISIBLE),
        CameraAngle.THREE_QUARTER: (Reliability.LOW, _NOTE_THREE_QUARTER),
        CameraAngle.UNKNOWN: (Reliability.LOW, None),
    },
    KNEE_BEND: {
        # Knee flexion happens in the sagittal plane — visible from almost any angle.
        CameraAngle.SIDE_LEFT: (Reliability.HIGH, None),
        CameraAngle.SIDE_RIGHT: (Reliability.HIGH, None),
        CameraAngle.FRONT: (Reliability.HIGH, None),
        CameraAngle.BACK: (Reliability.HIGH, None),
        CameraAngle.THREE_QUARTER: (Reliability.HIGH, None),
        CameraAngle.UNKNOWN: (Reliability.HIGH, None),
    },
    HEAD_STABILITY: {
        # Vertical jitter is preserved in any 2D projection.
        CameraAngle.SIDE_LEFT: (Reliability.HIGH, None),
        CameraAngle.SIDE_RIGHT: (Reliability.HIGH, None),
        CameraAngle.FRONT: (Reliability.HIGH, None),
        CameraAngle.BACK: (Reliability.HIGH, None),
        CameraAngle.THREE_QUARTER: (Reliability.HIGH, None),
        CameraAngle.UNKNOWN: (Reliability.HIGH, None),
    },
    SWING_DURATION: {
        # Time is invariant to camera angle.
        CameraAngle.SIDE_LEFT: (Reliability.HIGH, None),
        CameraAngle.SIDE_RIGHT: (Reliability.HIGH, None),
        CameraAngle.FRONT: (Reliability.HIGH, None),
        CameraAngle.BACK: (Reliability.HIGH, None),
        CameraAngle.THREE_QUARTER: (Reliability.HIGH, None),
        CameraAngle.UNKNOWN: (Reliability.HIGH, None),
    },
}


def lookup_reliability(feature: str, angle: CameraAngle) -> tuple[Reliability, str | None]:
    """Return (reliability, note) for a given feature + camera angle.

    Args:
        feature: One of the module-level feature name constants.
        angle: The camera angle the video was filmed from.

    Returns:
        Tuple of (Reliability bucket, optional human-readable note).

    Raises:
        KeyError: If `feature` is not in the reliability table. This is
            intentional — a typo or undeclared feature should fail loudly,
            not silently return UNRELIABLE.
    """
    if feature not in RELIABILITY_TABLE:
        raise KeyError(
            f"No reliability mapping for feature '{feature}'. "
            f"Known features: {sorted(RELIABILITY_TABLE.keys())}"
        )
    return RELIABILITY_TABLE[feature][angle]
