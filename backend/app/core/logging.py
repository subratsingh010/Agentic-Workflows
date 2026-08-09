import logging
import re
from typing import Any

import structlog


SECRET_PATTERNS = [
    re.compile(r"Bearer\s+[A-Za-z0-9._-]+"),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    re.compile(r"sk-[A-Za-z0-9_-]+"),
]


def redact(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    redacted = value
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def redaction_processor(_: Any, __: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    return {key: redact(value) for key, value in event_dict.items()}


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format="%(message)s")
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            redaction_processor,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )

