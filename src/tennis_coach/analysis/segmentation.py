"""Swing phase segmentation from pose keypoints.

A forehand groundstroke has four phases: ready, backswing, forward
swing, and follow-through. This module identifies the frame boundaries
of each phase using wrist velocity — the most robust signal because
peak velocity reliably coincides with ball contact regardless of stroke
variations.

Assumes a right-handed player (dominant wrist = RIGHT_WRIST).
Handedness detection is a TODO for v2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from loguru import logger
from scipy.signal import savgol_filter

from tennis_coach.vision.skeleton import RIGHT_WRIST


@dataclass(frozen=True)
class SwingPhases:
    """Frame indices marking the boundaries of each forehand phase.

    All values are frame indices into the keypoints array passed
    to `segment_forehand`. None for any field means that phase
    boundary couldn't be reliably detected.
    """

    backswing_start: int | None
    contact: int
    followthrough_end: int | None

    @property
    def ready_end(self) -> int | None:
        """Last frame of the ready phase (== backswing_start)."""
        return self.backswing_start


def segment_forehand(
    keypoints: np.ndarray,
    fps: float = 30.0,
    smoothing_window: int = 7,
) -> SwingPhases:
    """Identify swing phase boundaries from keypoint trajectories.

    Args:
        keypoints: Array of shape (num_frames, 33, 4) from extract_pose.
        fps: Source video frame rate. Used for velocity scaling and logging.
        smoothing_window: Savitzky-Golay window size for velocity smoothing.
            Must be odd and >= 5. Larger = smoother but blurrier event timing.

    Returns:
        SwingPhases with frame indices for backswing start, contact, and
        follow-through end.

    Raises:
        ValueError: If keypoints array is too short or has no detected poses.
    """
    if len(keypoints) < smoothing_window * 2:
        raise ValueError(
            f"Video too short for segmentation: {len(keypoints)} frames "
            f"(need at least {smoothing_window * 2})"
        )

    # 1. Extract right wrist trajectory (x, y over time).
    wrist_x = keypoints[:, RIGHT_WRIST, 0]
    wrist_y = keypoints[:, RIGHT_WRIST, 1]

    # 2. Handle missing detections by interpolating linearly.
    wrist_x = _fill_nans(wrist_x)
    wrist_y = _fill_nans(wrist_y)

    # 3. Smooth the trajectory before differentiating.
    #    Raw position differentiation amplifies noise catastrophically.
    wrist_x_smooth = savgol_filter(wrist_x, smoothing_window, polyorder=3)
    wrist_y_smooth = savgol_filter(wrist_y, smoothing_window, polyorder=3)

    # 4. Compute frame-to-frame velocity (magnitude).
    vx = np.diff(wrist_x_smooth, prepend=wrist_x_smooth[0])
    vy = np.diff(wrist_y_smooth, prepend=wrist_y_smooth[0])
    speed = np.sqrt(vx**2 + vy**2)

    # 5. Contact = global peak of wrist speed.
    contact = int(np.argmax(speed))
    logger.info("Contact detected at frame {} (speed={:.4f})", contact, speed[contact])

    # 6. Backswing start = last local minimum of speed before contact.
    backswing_start = _last_local_min_before(speed, contact)

    # 7. Follow-through end = first frame after contact where speed
    #    drops below 20% of peak.
    threshold = speed[contact] * 0.2
    after_contact = speed[contact + 1 :]
    below = np.where(after_contact < threshold)[0]
    followthrough_end = int(contact + 1 + below[0]) if len(below) > 0 else None

    phases = SwingPhases(
        backswing_start=backswing_start,
        contact=contact,
        followthrough_end=followthrough_end,
    )
    _log_phases(phases, len(keypoints), fps)
    return phases


def _fill_nans(arr: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaN values in a 1D array. Edges use nearest valid."""
    if not np.isnan(arr).any():
        return arr
    out = arr.copy()
    nan_mask = np.isnan(out)
    valid_idx = np.where(~nan_mask)[0]
    if len(valid_idx) == 0:
        raise ValueError("No valid keypoint values to interpolate from")
    out[nan_mask] = np.interp(np.where(nan_mask)[0], valid_idx, out[valid_idx])
    return out


def _last_local_min_before(signal: np.ndarray, end_idx: int) -> int | None:
    """Find the last local minimum of `signal` at indices < end_idx."""
    if end_idx < 2:
        return None
    # Local min: lower than both neighbors.
    candidates = []
    for i in range(1, end_idx):
        if signal[i] < signal[i - 1] and signal[i] < signal[i + 1]:
            candidates.append(i)
    return candidates[-1] if candidates else None


def _log_phases(phases: SwingPhases, total_frames: int, fps: float) -> None:
    """Pretty-print detected phases with both frame indices and timestamps."""

    def fmt(f: int | None) -> str:
        return f"frame {f} ({f / fps:.2f}s)" if f is not None else "not detected"

    logger.info(
        "Swing phases — backswing: {} | contact: {} | followthrough end: {}",
        fmt(phases.backswing_start),
        fmt(phases.contact),
        fmt(phases.followthrough_end),
    )
