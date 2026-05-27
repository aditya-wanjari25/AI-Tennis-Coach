"""Pose extraction from video using MediaPipe Tasks API.

Wraps MediaPipe's PoseLandmarker in a clean interface that takes a
video path and returns a numpy array of keypoints. Designed to be the
entry point of the vision pipeline.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from loguru import logger
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from tennis_coach.config import settings

# MediaPipe Pose has 33 landmarks per detected person.
NUM_LANDMARKS = 33
# Each landmark has 4 values: x, y, z, visibility.
LANDMARK_DIMS = 4


def extract_pose(video_path: Path) -> np.ndarray:
    """Run MediaPipe Pose on every frame of a video.

    Args:
        video_path: Path to an input video file (.mp4, .mov, etc.).

    Returns:
        Array of shape (num_frames, 33, 4). Each landmark is
        (x, y, z, visibility). x and y are normalized to [0, 1]
        relative to image dimensions. Frames where no pose was
        detected are filled with NaN.

    Raises:
        FileNotFoundError: If video_path or model file does not exist.
        RuntimeError: If the video cannot be opened by OpenCV.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not settings.pose_model_path.exists():
        raise FileNotFoundError(
            f"Pose model not found at {settings.pose_model_path}. "
            "Download with the curl command in README."
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(
        "Extracting pose from {} ({} frames @ {:.1f} fps)",
        video_path.name,
        total_frames,
        fps,
    )

    keypoints = np.full(
        (total_frames, NUM_LANDMARKS, LANDMARK_DIMS),
        fill_value=np.nan,
        dtype=np.float32,
    )

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(settings.pose_model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        min_pose_detection_confidence=settings.pose_min_detection_confidence,
        min_tracking_confidence=settings.pose_min_tracking_confidence,
    )

    frame_idx = 0
    detected_frames = 0
    with mp_vision.PoseLandmarker.create_from_options(options) as landmarker:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # MediaPipe expects RGB; OpenCV loads BGR.
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                # Video mode requires a monotonically increasing timestamp (ms).
                timestamp_ms = int((frame_idx / fps) * 1000) if fps > 0 else frame_idx
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if result.pose_landmarks:
                    # Take the first detected person (we'll add player selection later).
                    landmarks = result.pose_landmarks[0]
                    for i, lm in enumerate(landmarks):
                        keypoints[frame_idx, i] = [lm.x, lm.y, lm.z, lm.visibility]
                    detected_frames += 1

                frame_idx += 1
        finally:
            cap.release()

    detection_rate = detected_frames / frame_idx if frame_idx > 0 else 0.0
    logger.info(
        "Pose extracted: {}/{} frames had a detected pose ({:.1%})",
        detected_frames,
        frame_idx,
        detection_rate,
    )

    if detection_rate < 0.5:
        logger.warning(
            "Low detection rate ({:.1%}). Check camera angle, lighting, "
            "and that the player is fully in frame.",
            detection_rate,
        )

    return keypoints[:frame_idx]


def save_keypoints(keypoints: np.ndarray, path: Path) -> None:
    """Save extracted keypoints to disk as a NumPy `.npy` file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, keypoints)
    logger.info("Saved keypoints to {} (shape {})", path, keypoints.shape)


def load_keypoints(path: Path) -> np.ndarray:
    """Load previously-extracted keypoints from a `.npy` file."""
    if not path.exists():
        raise FileNotFoundError(f"Keypoints file not found: {path}")
    keypoints: np.ndarray = np.load(path)
    logger.info("Loaded keypoints from {} (shape {})", path, keypoints.shape)
    return keypoints
