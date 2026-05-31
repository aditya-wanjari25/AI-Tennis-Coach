"""Single configured LLM client for the coaching agent.

Centralized so model choice, temperature, and retry config live in
one place. All nodes import `get_llm()` rather than instantiating
ChatAnthropic directly.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_anthropic import ChatAnthropic
from loguru import logger

from tennis_coach.config import settings


@lru_cache(maxsize=1)
def get_llm() -> ChatAnthropic:
    """Return a configured ChatAnthropic instance (cached).

    Cached so we don't reinstantiate the client on every node call.
    Temperature is low — we want consistent, structured coaching, not
    creative writing.
    """
    logger.info("Initializing ChatAnthropic with model={}", settings.anthropic_model)
    return ChatAnthropic(
        model_name=settings.anthropic_model,
        api_key=settings.anthropic_api_key,
        temperature=0.3,
        max_tokens=1024,
        timeout=30.0,
        max_retries=2,
    )
