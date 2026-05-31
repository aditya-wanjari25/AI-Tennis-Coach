"""Core types shared across the analysis pipeline.

Defined once here so every module references the same vocabulary.
Adding a new camera angle, handedness, or reliability bucket happens
in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CameraAngle(str, Enum):
    """Direction the camera is facing relative to the player.

    Values describe what the camera *sees*:
      - SIDE_LEFT/SIDE_RIGHT: camera on the player's left/right.
        Player's hitting plane is perpendicular to the lens — ideal.
      - FRONT: camera in front of the player (player faces lens).
      - BACK: camera behind the player.
      - THREE_QUARTER: angled view, neither pure side nor pure front/back.
      - UNKNOWN: angle could not be determined or not provided.
    """

    SIDE_LEFT = "side_left"
    SIDE_RIGHT = "side_right"
    FRONT = "front"
    BACK = "back"
    THREE_QUARTER = "three_quarter"
    UNKNOWN = "unknown"


class Handedness(str, Enum):
    """Which hand the player uses to hit groundstrokes."""

    RIGHT = "right"
    LEFT = "left"
    UNKNOWN = "unknown"


class Reliability(str, Enum):
    """How much to trust a given measurement.

    Used by the coaching layer to decide whether to surface a
    measurement and how strongly to phrase the feedback.
    """

    HIGH = "high"
    LOW = "low"
    UNRELIABLE = "unreliable"


@dataclass(frozen=True)
class Measurement:
    """A single biomechanical measurement with reliability context.

    Replaces bare `float | None` values in feature outputs. Forces
    every consumer to think about reliability, not just value.

    Attributes:
        value: The measured number. None if measurement couldn't be
            computed (e.g. missing keypoints).
        reliability: How much to trust the value, given camera angle.
        unit: Display unit for the value (e.g. "°", "ms").
        note: Optional human-readable caveat (e.g. "back view limits
            depth measurement"). Surfaced in coaching prompts.
    """

    value: float | None
    reliability: Reliability
    unit: str = ""
    note: str | None = None

    @property
    def is_usable(self) -> bool:
        """True if both a value exists AND reliability is HIGH or LOW."""
        return self.value is not None and self.reliability is not Reliability.UNRELIABLE
