"""Prompt templates for the coaching LLM.

Kept isolated from logic so prompt iteration doesn't require touching
graph or node code. Each prompt is a callable that takes structured
inputs and returns a (system_prompt, user_prompt) tuple.
"""

from __future__ import annotations

from tennis_coach.analysis.classification import ClassificationResult
from tennis_coach.analysis.features import SwingFeatures
from tennis_coach.analysis.types import CameraAngle, Handedness

COACH_SYSTEM_PROMPT = """\
You are a tennis coach analyzing a player's forehand groundstroke. \
A computer-vision pipeline has already extracted biomechanical \
measurements and a separate analysis layer has classified which \
measurements represent real issues. You are given ONLY the issues \
the analysis layer flagged.

Your job is to give honest, actionable, encouraging feedback. You write \
like a real coach speaking to a player — direct, warm, never patronizing.

Strict rules you always follow:
- Coach ONLY on the issues listed under "Flagged findings". \
  Do NOT infer or invent additional issues from any reference data shown.
- If reference measurements are shown, treat them as context only — \
  do not coach on them unless they appear under "Flagged findings".
- Prioritize findings by severity: address major issues first, then \
  moderate, then minor.
- Always include exactly ONE concrete drill suggestion the player can \
  do in their next practice. The drill should target the highest-severity \
  finding.
- Keep the total response to 200-350 words. No preambles, no sign-offs.
- Use markdown: a short opening summary, then a "## What to work on" \
  section with prioritized bullets, then a "## Drill for next session" section.
"""


def build_issues_prompt(
    features: SwingFeatures,
    classification: ClassificationResult,
    camera_angle: CameraAngle,
    handedness: Handedness,
) -> tuple[str, str]:
    """Construct (system, user) prompt for the 'issues found' branch."""
    user_prompt = _format_user_context(features, classification, camera_angle, handedness)
    user_prompt += (
        "\n\n"
        "Write a coaching response addressing the flagged findings above. "
        "Lead with the most severe issue. Do not introduce concerns about "
        "measurements that were not flagged."
    )
    return COACH_SYSTEM_PROMPT, user_prompt


def _format_user_context(
    features: SwingFeatures,
    classification: ClassificationResult,
    camera_angle: CameraAngle,
    handedness: Handedness,
) -> str:
    """Render the structured swing data — findings are primary, measurements secondary."""
    lines = [
        "## Player setup",
        f"- Handedness: {handedness.value}",
        f"- Camera angle: {camera_angle.value.replace('_', ' ')}",
        f"- {classification.reliability_summary}",
        "",
    ]

    # Flagged findings come FIRST and are the primary payload.
    if classification.findings:
        lines.append("## Flagged findings (the ONLY issues to coach on)")
        for f in sorted(classification.findings, key=lambda x: _severity_order(x.severity)):
            lines.append(f"- [{f.severity}] {f.observation}")
    else:
        lines.append("## Flagged findings")
        lines.append("(none — no measurements fell outside healthy ranges)")

    # Raw measurements as reference only, clearly subordinated.
    lines.append("")
    lines.append("## Reference measurements (context only — do NOT coach on these)")
    feature_items = [
        ("Elbow angle at contact", features.elbow_angle_at_contact),
        ("Contact height vs shoulder", features.contact_height_vs_shoulder),
        ("Hip-shoulder separation", features.hip_shoulder_separation),
        ("Knee bend at contact", features.knee_bend_at_contact),
        ("Head stability (Y std-dev)", features.head_stability),
        ("Swing duration", features.swing_duration_ms),
    ]
    for label, m in feature_items:
        if m.value is None:
            lines.append(f"- {label}: not measured")
            continue
        val_str = f"{m.value:.2f}{m.unit}"
        lines.append(f"- {label}: {val_str} [{m.reliability.value}]")

    return "\n".join(lines)


def _severity_order(severity: str) -> int:
    """For sorting findings by importance — major first."""
    return {"major": 0, "moderate": 1, "minor": 2}.get(severity, 99)


def build_healthy_prompt(
    features: SwingFeatures,
    classification: ClassificationResult,
    camera_angle: CameraAngle,
    handedness: Handedness,
) -> tuple[str, str]:
    """Construct (system, user) prompt for the 'healthy swing' branch.

    Used when classification returned no findings. Tone is confirming
    and forward-looking rather than diagnostic.
    """
    system_prompt = """\
You are a tennis coach analyzing a player's forehand. The biomechanical \
analysis found no measurable issues — all reliable measurements fell \
within healthy ranges.

Your job: confirm the player's form is solid, then suggest ONE concrete \
focus area for their next session that pushes their game forward without \
manufacturing problems that don't exist.

Strict rules:
- Do NOT invent issues. The analysis found no problems and you must not \
  imply otherwise.
- Acknowledge any camera-angle limitations honestly if reliability was \
  reduced — it's fine to say "from this angle we can confirm X, Y, Z."
- 150-250 words. No preamble, no sign-off.
- Use markdown: a short confirming summary, then "## Focus area for next \
  session" with one specific drill or skill to practice.
"""
    user_prompt = _format_user_context(features, classification, camera_angle, handedness)
    user_prompt += (
        "\n\n"
        "The swing looks good. Confirm what's working and suggest one "
        "focus area to keep developing."
    )
    return system_prompt, user_prompt


def build_insufficient_data_prompt(
    features: SwingFeatures,
    classification: ClassificationResult,
    camera_angle: CameraAngle,
    handedness: Handedness,
) -> tuple[str, str]:
    """Construct (system, user) prompt for the 'insufficient data' branch.

    Used when too few reliable measurements were available. The response
    should explain the limitation and guide the player toward better footage.
    """
    system_prompt = """\
You are a tennis coach reviewing a player's forehand video. The \
biomechanical analysis was unable to extract enough reliable \
measurements to give meaningful technical feedback — typically because \
of camera angle, framing, lighting, or footage length.

Your job: explain the limitation kindly, tell the player exactly how \
to capture better footage, and (if any measurements WERE reliable) \
acknowledge what was learned from those.

Strict rules:
- Do NOT speculate about technique based on unreliable measurements.
- Be concrete about how to re-film: angle, distance, lighting, \
  clip length, what should be in frame.
- 150-250 words. No preamble, no sign-off.
- Use markdown: a short explanation, then "## How to capture better \
  footage" with specific steps.
"""
    user_prompt = _format_user_context(features, classification, camera_angle, handedness)
    user_prompt += (
        "\n\n"
        "Explain why technical analysis is limited and tell the player "
        "how to re-film for a complete assessment."
    )
    return system_prompt, user_prompt
