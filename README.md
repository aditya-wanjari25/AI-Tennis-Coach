# AI Tennis Coach

An AI-powered tennis coach. Drop in a video of yourself swinging,
get back a structured coaching report grounded in computer vision and
synthesized by Claude.

This is a personal project built to explore the boundary between
deterministic computer vision and LLM-based reasoning: pose extraction
and biomechanical math produce *measurements*; an LLM produces *coaching*.
The two are deliberately kept separate so the LLM can never invent
metrics — only interpret them.

## How it works
Video (.mp4)
│
▼
MediaPipe Pose ──────────►  Keypoints (frames × 33 landmarks × 4)
│
▼
Swing segmentation ──────►  Backswing / contact / follow-through frames
│
▼
Biomechanical features ──►  Elbow angle, knee bend, head stability, ...
│                       (each tagged with a reliability rating based
│                        on camera angle)
▼
Rule-based classification─►  HEALTHY  /  ISSUES  /  INSUFFICIENT_DATA
│
▼
LangGraph branches ───────►  Routed prompt → Claude → coaching markdown

The vision pipeline (MediaPipe + OpenCV + NumPy) is fully deterministic.
The LLM only sees structured findings — never raw measurements alone —
which structurally prevents it from coaching on things the analysis
didn't actually flag.

## Quick start

Prerequisites: Python 3.11+, [uv](https://docs.astral.sh/uv/),
an Anthropic API key.

```bash
# 1. Clone and enter the project
git clone https://github.com/<you>/ai-tennis-coach.git
cd ai-tennis-coach

# 2. Install everything (uv creates the venv automatically)
uv sync

# 3. Download the MediaPipe pose model (~9 MB)
mkdir -p models
curl -L -o models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task

# 4. Configure your API key and player setup
cp .env.example .env
# then edit .env: set ANTHROPIC_API_KEY, CAMERA_ANGLE, HANDEDNESS

# 5. Drop a forehand clip into data/
# (side-on camera angle gives the best results)
mv ~/Movies/my_forehand.mp4 data/forehand_01.mp4

# 6. Run the pipeline
uv run python -m tennis_coach.scripts.analyze data/forehand_01.mp4
```

## Tech stack

- **Package management**: uv
- **Computer vision**: MediaPipe (Tasks API) + OpenCV + NumPy + SciPy
- **Agent**: LangGraph
- **LLM**: Anthropic Claude (via `langchain-anthropic`)
- **Config**: pydantic-settings
- **Logging**: loguru
- **Quality**: ruff + pytest + pre-commit
