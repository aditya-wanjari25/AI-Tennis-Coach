"""Centralized logging configuration using loguru.

Loguru is configured once at application startup via `setup_logging()`.
All other modules simply `from loguru import logger` and use it directly —
they don't need to know about handlers, formatters, or routing.

Stdlib `logging` calls from third-party libraries (MediaPipe, OpenCV,
LangChain) are intercepted and re-routed through loguru so all logs
flow through a single pipe with consistent formatting.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Record


class InterceptHandler(logging.Handler):
    """Routes stdlib `logging` records into loguru.

    Third-party libraries (e.g. MediaPipe, LangChain) use stdlib logging.
    Without this handler, their logs would bypass loguru entirely and
    appear with different formatting (or vanish silently).
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Map stdlib level number to loguru level name, falling back to the number.
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find the caller frame so the log shows the original call site,
        # not this intercept handler.
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__ and frame.f_back:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _format_record(record: Record) -> str:
    """Custom format string for terminal logs.

    Includes timestamp, level, module:function:line, and the message.
    Colors are applied via loguru's `<color>` tags when output is a TTY.
    """
    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>\n{exception}"
    )


def setup_logging(level: str = "INFO") -> None:
    """Configure loguru and intercept stdlib logging.

    Called once at application startup, typically from the entrypoint
    or the LangGraph builder.

    Args:
        level: Minimum log level to emit. One of TRACE, DEBUG, INFO,
            SUCCESS, WARNING, ERROR, CRITICAL. Read from settings.log_level
            in production.
    """
    # Remove loguru's default handler so we control the output entirely.
    logger.remove()

    # Add our configured terminal handler.
    logger.add(
        sys.stderr,
        level=level.upper(),
        format=_format_record,
        colorize=True,
        backtrace=True,  # show full stack on exceptions
        diagnose=True,  # show variable values in tracebacks (dev only)
    )

    # Route all stdlib logging through loguru.
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # Silence noisy third-party loggers we don't care about.
    for noisy in ("urllib3", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.debug("Logging configured at level {}", level.upper())
