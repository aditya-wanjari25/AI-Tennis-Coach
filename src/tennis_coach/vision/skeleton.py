"""Skeleton overlay rendering on video frames.

Takes raw keypoints from `extract_pose` and renders them back onto
the source video as a colored skeleton — circles for joints, lines
for bones. Used both for debugging the pose pipeline and as the
foundation for the "correction overlay" feature later.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from loguru import logger

# MediaPipe Pose landmark indices.
# Full list: https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
NOSE = 0
LEFT_EYE_INNER, LEFT_EYE, LEFT_EYE_OUTER = 1, 2, 3
RIGHT_EYE_INNER, RIGHT_EYE, RIGHT_EYE_OUTER = 4, 5, 6
LEFT_EAR, RIGHT_EAR = 7, 8
MOUTH_LEFT, MOUTH_RIGHT = 9, 10
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_PINKY, RIGHT_PINKY = 17, 18
LEFT_INDEX, RIGHT_INDEX = 19, 20
LEFT_THUMB, RIGHT_THUMB = 21, 22
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28
LEFT_HEEL, RIGHT_HEEL = 29, 30
LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX = 31, 32

# Bones to draw (pairs of landmark indices).
SKELETON_CONNECTIONS: list[tuple[int, int]] = [
    # Torso
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    # Left arm
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    # Right arm
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    # Left leg
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    # Right leg
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
]

# Colors in BGR (OpenCV convention).
COLOR_JOINT = (0, 255, 0)  # green dots
COLOR_BONE = (255, 200, 0)  # cyan-ish lines
COLOR_LOW_CONF = (0, 0, 255)  # red — drawn when visibility < threshold

VISIBILITY_THRESHOLD = 0.5


def render_skeleton_video(
    input_video: Path,
    keypoints: np.ndarray,
    output_video: Path,
) -> None:
    """Render skeleton overlay onto a video and save the result.

    Args:
        input_video: Path to source video.
        keypoints: Array of shape (num_frames, 33, 4) from extract_pose.
        output_video: Path where the overlaid MP4 will be written.

    Raises:
        RuntimeError: If video I/O fails.
    """
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_video}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video for writing: {output_video}")

    logger.info(
        "Rendering skeleton overlay → {} ({}x{} @ {:.1f} fps)",
        output_video.name,
        width,
        height,
        fps,
    )

    frame_idx = 0
    try:
        while frame_idx < len(keypoints):
            ret, frame = cap.read()
            if not ret:
                break
            _draw_skeleton_on_frame(frame, keypoints[frame_idx], width, height)
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    logger.info("Wrote {} frames to {}", frame_idx, output_video)


def _draw_skeleton_on_frame(
    frame: np.ndarray,
    frame_keypoints: np.ndarray,
    width: int,
    height: int,
) -> None:
    """Mutate `frame` in place, drawing the skeleton from `frame_keypoints`.

    Args:
        frame: BGR image to draw on.
        frame_keypoints: Array of shape (33, 4) — landmarks for this frame.
        width: Frame width in pixels (for denormalizing x).
        height: Frame height in pixels (for denormalizing y).
    """
    # If this frame had no detection, all values are NaN — skip silently.
    if np.isnan(frame_keypoints).all():
        return

    # Draw bones first, joints on top.
    for start_idx, end_idx in SKELETON_CONNECTIONS:
        start = frame_keypoints[start_idx]
        end = frame_keypoints[end_idx]
        if np.isnan(start).any() or np.isnan(end).any():
            continue

        color = COLOR_BONE if min(start[3], end[3]) >= VISIBILITY_THRESHOLD else COLOR_LOW_CONF
        x1, y1 = int(start[0] * width), int(start[1] * height)
        x2, y2 = int(end[0] * width), int(end[1] * height)
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness=2)

    # Draw joints.
    for i in range(33):
        lm = frame_keypoints[i]
        if np.isnan(lm).any():
            continue
        color = COLOR_JOINT if lm[3] >= VISIBILITY_THRESHOLD else COLOR_LOW_CONF
        x, y = int(lm[0] * width), int(lm[1] * height)
        cv2.circle(frame, (x, y), radius=4, color=color, thickness=-1)
