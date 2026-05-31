"""LangGraph coaching agent — assembles the swing-analysis pipeline.

The graph orchestrates: classification → prompt selection → LLM call.
State flows through typed nodes; branching is determined by the
classification result.
"""

from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from loguru import logger

from tennis_coach.analysis.classification import (
    ClassificationResult,
    SwingClassification,
    classify_swing,
)
from tennis_coach.analysis.features import SwingFeatures
from tennis_coach.analysis.types import CameraAngle, Handedness
from tennis_coach.feedback.llm import get_llm
from tennis_coach.feedback.prompts import (
    build_healthy_prompt,
    build_insufficient_data_prompt,
    build_issues_prompt,
)


class CoachingState(TypedDict):
    """State that flows through the coaching graph.

    Inputs are set before the graph runs; intermediate and output
    fields are populated by nodes.
    """

    # Inputs
    features: SwingFeatures
    camera_angle: CameraAngle
    handedness: Handedness

    # Set by classify_node
    classification: ClassificationResult

    # Set by one of the three feedback nodes
    coaching_text: str


def classify_node(state: CoachingState) -> dict:
    """Run rule-based classification on the features."""
    result = classify_swing(state["features"])
    logger.info(
        "Classification: {} ({} findings)",
        result.classification.value,
        len(result.findings),
    )
    return {"classification": result}


def issues_feedback_node(state: CoachingState) -> dict:
    """Call Claude with the issues prompt."""
    logger.info("Generating coaching feedback (issues branch)")
    system_p, user_p = build_issues_prompt(
        state["features"],
        state["classification"],
        state["camera_angle"],
        state["handedness"],
    )
    return _invoke_llm(system_p, user_p)


def healthy_feedback_node(state: CoachingState) -> dict:
    """Call Claude with the healthy-swing prompt."""
    logger.info("Generating coaching feedback (healthy branch)")
    system_p, user_p = build_healthy_prompt(
        state["features"],
        state["classification"],
        state["camera_angle"],
        state["handedness"],
    )
    return _invoke_llm(system_p, user_p)


def insufficient_data_feedback_node(state: CoachingState) -> dict:
    """Call Claude with the insufficient-data prompt."""
    logger.info("Generating coaching feedback (insufficient-data branch)")
    system_p, user_p = build_insufficient_data_prompt(
        state["features"],
        state["classification"],
        state["camera_angle"],
        state["handedness"],
    )
    return _invoke_llm(system_p, user_p)


def _invoke_llm(system_p: str, user_p: str) -> dict:
    """Shared LLM invocation: call Claude, return state update."""
    llm = get_llm()
    response = llm.invoke([SystemMessage(content=system_p), HumanMessage(content=user_p)])
    text = response.content if isinstance(response.content, str) else str(response.content)
    return {"coaching_text": text}


def route_after_classification(state: CoachingState) -> str:
    """Pick which feedback node to run based on the classification."""
    cls = state["classification"].classification
    if cls is SwingClassification.HEALTHY:
        return "healthy_feedback"
    if cls is SwingClassification.ISSUES:
        return "issues_feedback"
    return "insufficient_data_feedback"


def build_coaching_graph():
    """Build and compile the coaching graph."""
    graph = StateGraph(CoachingState)

    graph.add_node("classify", classify_node)
    graph.add_node("issues_feedback", issues_feedback_node)
    graph.add_node("healthy_feedback", healthy_feedback_node)
    graph.add_node("insufficient_data_feedback", insufficient_data_feedback_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges(
        "classify",
        route_after_classification,
        {
            "issues_feedback": "issues_feedback",
            "healthy_feedback": "healthy_feedback",
            "insufficient_data_feedback": "insufficient_data_feedback",
        },
    )
    graph.add_edge("issues_feedback", END)
    graph.add_edge("healthy_feedback", END)
    graph.add_edge("insufficient_data_feedback", END)

    return graph.compile()
