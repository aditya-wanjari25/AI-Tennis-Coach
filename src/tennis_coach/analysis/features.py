"""Biomechanical feature extraction from pose keypoints.

Computes the measurable quantities that tennis coaches use to evaluate
form: joint angles, contact height, body rotation, and tempo. Operates
on a single forehand swing — caller is responsible for selecting one
swing's keypoints + phase indices.

All angle outputs are in degrees. All positional outputs are in
normalized image coordinates (0-1), so they're resolution-independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tennis_coach.analysis.segmentation import SwingPhases
from tennis_coach.vision.skeleton import (
    LEFT_HIP,
    LEFT_SHOULDER,
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
    """Per-swing biomechanical measurements.

    All values may be None if the underlying keypoints had missing
    detections in the relevant frames. Downstream consumers must
    handle None.
    """

    elbow_angle_at_contact: float | None
    contact_height_vs_shoulder: float | None  # +ve = above shoulder, -ve = below
    hip_shoulder_separation: float | None
    knee_bend_at_contact: float | None
    head_stability: float | None  # std-dev of nose Y across swing, lower = more stable
    swing_duration_ms: float | None


def extract_features(
    keypoints: np.ndarray,
    phases: SwingPhases,
    fps: float = 30.0,
) -> SwingFeatures:
    """Compute biomechanical features for a single forehand swing.

    Args:
        keypoints: Array of shape (num_frames, 33, 4) from extract_pose.
        phases: Detected swing phases (must include `contact`).
        fps: Video frame rate, used to convert frame counts to milliseconds.

    Returns:
        SwingFeatures with each metric computed (or None where keypoints
        were missing).
    """
    contact = phases.contact
    contact_frame = keypoints[contact]

    # 1. Elbow angle at contact: angle at right elbow between shoulder & wrist.
    elbow_angle = _angle_between_points(
        contact_frame[RIGHT_SHOULDER],
        contact_frame[RIGHT_ELBOW],
        contact_frame[RIGHT_WRIST],
    )

    # 2. Contact height: wrist Y vs shoulder Y. Note: image Y grows downward,
    #    so we negate to get "positive = wrist above shoulder."
    contact_height = _safe_subtract(
        contact_frame[RIGHT_SHOULDER][1],
        contact_frame[RIGHT_WRIST][1],
    )

    # 3. Hip-shoulder separation: angle between shoulder-line and hip-line vectors.
    hip_shoulder_sep = _hip_shoulder_separation_angle(contact_frame)

    # 4. Knee bend at contact: angle at right knee between hip & ankle.
    knee_angle = _angle_between_points(
        contact_frame[RIGHT_HIP],
        contact_frame[RIGHT_KNEE],
        contact_frame[RIGHT_ANKLE],
    )

    # 5. Head stability: std-dev of nose Y across the whole detected swing window.
    swing_start = phases.backswing_start if phases.backswing_start is not None else 0
    swing_end = phases.followthrough_end if phases.followthrough_end is not None else len(keypoints)
    nose_y_series = keypoints[swing_start : swing_end + 1, NOSE, 1]
    valid_nose_y = nose_y_series[~np.isnan(nose_y_series)]
    head_stability = float(np.std(valid_nose_y)) if len(valid_nose_y) >= 2 else None

    # 6. Swing tempo: total duration in ms (only if both endpoints detected).
    if phases.backswing_start is not None and phases.followthrough_end is not None:
        swing_duration_ms: float | None = (
            (phases.followthrough_end - phases.backswing_start) / fps * 1000.0
        )
    else:
        swing_duration_ms = None

    return SwingFeatures(
        elbow_angle_at_contact=elbow_angle,
        contact_height_vs_shoulder=contact_height,
        hip_shoulder_separation=hip_shoulder_sep,
        knee_bend_at_contact=knee_angle,
        head_stability=head_stability,
        swing_duration_ms=swing_duration_ms,
    )


def _angle_between_points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float | None:
    """Compute the angle at point `b` formed by rays b→a and b→c, in degrees.

    Returns None if any of the three landmarks has a NaN coordinate.
    Uses only x,y components — z is unreliable from monocular video.
    """
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
    """Angle between the shoulder-line vector and the hip-line vector.

    A non-zero angle indicates the upper body is rotated relative to
    the hips — i.e. the "X-factor" of stored torsional energy.
    """
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
