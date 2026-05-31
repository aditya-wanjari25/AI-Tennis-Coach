"""Biomechanical feature extraction from pose keypoints.

Computes the measurable quantities that tennis coaches use to evaluate
form: joint angles, contact height, body rotation, and tempo.

Every feature returns a Measurement object that includes both the value
and a reliability rating for the given camera angle. The coaching layer
consults reliability to decide what to surface and how strongly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tennis_coach.analysis.reliability import (
    CONTACT_HEIGHT,
    ELBOW_ANGLE,
    HEAD_STABILITY,
    HIP_SHOULDER_SEP,
    KNEE_BEND,
    SWING_DURATION,
    lookup_reliability,
)
from tennis_coach.analysis.segmentation import SwingPhases
from tennis_coach.analysis.types import CameraAngle, Handedness, Measurement
from tennis_coach.vision.skeleton import (
    LEFT_ANKLE,
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_KNEE,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_ANKLE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_KNEE,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)


@dataclass(frozen=True)
class SwingFeatures:
    """Per-swing biomechanical measurements, each with reliability metadata."""

    elbow_angle_at_contact: Measurement
    contact_height_vs_shoulder: Measurement
    hip_shoulder_separation: Measurement
    knee_bend_at_contact: Measurement
    head_stability: Measurement
    swing_duration_ms: Measurement


def extract_features(
    keypoints: np.ndarray,
    phases: SwingPhases,
    camera_angle: CameraAngle,
    handedness: Handedness,
    fps: float = 30.0,
) -> SwingFeatures:
    """Compute biomechanical features for a single forehand swing.

    Args:
        keypoints: Array of shape (num_frames, 33, 4) from extract_pose.
        phases: Detected swing phases.
        camera_angle: Camera position relative to player. Drives reliability.
        handedness: Player's dominant hand. Selects which arm to measure.
        fps: Video frame rate.

    Returns:
        SwingFeatures with each metric as a Measurement.
    """
    contact_frame = keypoints[phases.contact]

    # Resolve which side's landmarks count as "dominant" given handedness.
    if handedness is Handedness.LEFT:
        dom_shoulder, dom_elbow, dom_wrist = LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST
        dom_hip, dom_knee, dom_ankle = LEFT_HIP, LEFT_KNEE, LEFT_ANKLE
    else:
        # RIGHT and UNKNOWN both default to right-side landmarks.
        dom_shoulder, dom_elbow, dom_wrist = RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST
        dom_hip, dom_knee, dom_ankle = RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE

    # 1. Elbow angle: angle at dominant elbow between shoulder & wrist.
    elbow_value = _angle_between_points(
        contact_frame[dom_shoulder],
        contact_frame[dom_elbow],
        contact_frame[dom_wrist],
    )
    elbow = _make_measurement(elbow_value, ELBOW_ANGLE, camera_angle, unit="°")

    # 2. Contact height: positive value = wrist above shoulder.
    #    (Image Y grows downward, so we subtract wrist_y from shoulder_y.)
    contact_height_value = _safe_subtract(
        contact_frame[dom_shoulder][1],
        contact_frame[dom_wrist][1],
    )
    contact_height = _make_measurement(contact_height_value, CONTACT_HEIGHT, camera_angle)

    # 3. Hip-shoulder separation: rotation between hip and shoulder lines.
    hip_shoulder_value = _hip_shoulder_separation_angle(contact_frame)
    hip_shoulder = _make_measurement(hip_shoulder_value, HIP_SHOULDER_SEP, camera_angle, unit="°")

    # 4. Knee bend: angle at dominant knee between hip & ankle.
    knee_value = _angle_between_points(
        contact_frame[dom_hip],
        contact_frame[dom_knee],
        contact_frame[dom_ankle],
    )
    knee = _make_measurement(knee_value, KNEE_BEND, camera_angle, unit="°")

    # 5. Head stability: nose Y std-dev across the swing window.
    swing_start = phases.backswing_start if phases.backswing_start is not None else 0
    swing_end = phases.followthrough_end if phases.followthrough_end is not None else len(keypoints)
    nose_y = keypoints[swing_start : swing_end + 1, NOSE, 1]
    valid_nose_y = nose_y[~np.isnan(nose_y)]
    head_value = float(np.std(valid_nose_y)) if len(valid_nose_y) >= 2 else None
    head = _make_measurement(head_value, HEAD_STABILITY, camera_angle)

    # 6. Swing duration in ms.
    if phases.backswing_start is not None and phases.followthrough_end is not None:
        duration_value: float | None = (
            (phases.followthrough_end - phases.backswing_start) / fps * 1000.0
        )
    else:
        duration_value = None
    duration = _make_measurement(duration_value, SWING_DURATION, camera_angle, unit=" ms")

    return SwingFeatures(
        elbow_angle_at_contact=elbow,
        contact_height_vs_shoulder=contact_height,
        hip_shoulder_separation=hip_shoulder,
        knee_bend_at_contact=knee,
        head_stability=head,
        swing_duration_ms=duration,
    )


def _make_measurement(
    value: float | None,
    feature_name: str,
    camera_angle: CameraAngle,
    unit: str = "",
) -> Measurement:
    """Wrap a raw value with its reliability metadata."""
    reliability, note = lookup_reliability(feature_name, camera_angle)
    return Measurement(value=value, reliability=reliability, unit=unit, note=note)


# ─── Geometry helpers (pure functions, unchanged math) ───


def _angle_between_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    """Angle at point `b` formed by rays b→a and b→c, in degrees. None on NaN."""
    if np.isnan(a[:2]).any() or np.isnan(b[:2]).any() or np.isnan(c[:2]).any():
        return None
    ba = a[:2] - b[:2]
    bc = c[:2] - b[:2]
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return None
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def _hip_shoulder_separation_angle(frame: np.ndarray) -> float | None:
    """Angle between shoulder-line and hip-line vectors."""
    if np.isnan(frame[[LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP], :2]).any():
        return None
    shoulder_vec = frame[RIGHT_SHOULDER, :2] - frame[LEFT_SHOULDER, :2]
    hip_vec = frame[RIGHT_HIP, :2] - frame[LEFT_HIP, :2]
    cos_angle = np.dot(shoulder_vec, hip_vec) / (
        np.linalg.norm(shoulder_vec) * np.linalg.norm(hip_vec)
    )
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def _safe_subtract(a: float, b: float) -> float | None:
    if np.isnan(a) or np.isnan(b):
        return None
    return float(a - b)
