from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

try:
    from prometheus_client import Counter, Gauge, Histogram
except Exception:  # pragma: no cover - metrics are optional in minimal envs
    Counter = Gauge = Histogram = None  # type: ignore[assignment]


def _counter(name: str, description: str, labels: list[str]):
    if Counter is None:
        return None
    return Counter(name, description, labels)


def _histogram(name: str, description: str, labels: list[str], buckets: tuple[float, ...]):
    if Histogram is None:
        return None
    return Histogram(name, description, labels, buckets=buckets)


def _gauge(name: str, description: str, labels: list[str]):
    if Gauge is None:
        return None
    return Gauge(name, description, labels)


RAG_RETRIEVAL_SECONDS = _histogram(
    "employee_support_rag_retrieval_seconds",
    "Policy RAG retrieval latency in seconds.",
    ["backend", "mode"],
    (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
RAG_RERANK_SECONDS = _histogram(
    "employee_support_rag_rerank_seconds",
    "Policy RAG rerank latency in seconds.",
    ["provider"],
    (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2),
)
LLM_SECONDS = _histogram(
    "employee_support_llm_seconds",
    "LLM generation latency in seconds.",
    ["provider", "model"],
    (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30),
)
EVAL_SECONDS = _histogram(
    "employee_support_eval_seconds",
    "Policy evaluation run latency in seconds.",
    ["backend", "mode"],
    (0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
)
AGENT_NODE_SECONDS = _histogram(
    "employee_support_agent_node_seconds",
    "Single-agent LangGraph node latency in seconds.",
    ["node", "intent"],
    (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
RAG_RETRIEVED_CHUNKS = _histogram(
    "employee_support_rag_retrieved_chunks",
    "Number of chunks returned by RAG retrieval.",
    ["backend", "mode"],
    (0, 1, 2, 3, 5, 8, 13, 20, 40, 80),
)
RAG_ERRORS = _counter(
    "employee_support_rag_errors_total",
    "RAG retrieval errors by backend and stage.",
    ["backend", "stage"],
)
LLM_ERRORS = _counter(
    "employee_support_llm_errors_total",
    "LLM generation errors by provider and model.",
    ["provider", "model"],
)
SECURITY_BLOCKS = _counter(
    "employee_support_security_blocks_total",
    "Security guardrail blocks by reason.",
    ["reason"],
)
EVAL_SCORE = _gauge(
    "employee_support_eval_score",
    "Latest policy evaluation score by metric.",
    ["metric"],
)
LLM_OUTPUT_TOKENS = _histogram(
    "employee_support_llm_output_tokens",
    "Approximate LLM output tokens/words by provider and model.",
    ["provider", "model"],
    (0, 16, 32, 64, 128, 256, 512, 1024),
)


def observe_histogram(metric, labels: dict[str, str], value: float) -> None:
    if metric is not None:
        metric.labels(**labels).observe(value)


def increment(metric, labels: dict[str, str]) -> None:
    if metric is not None:
        metric.labels(**labels).inc()


def set_gauge(metric, labels: dict[str, str], value: float) -> None:
    if metric is not None:
        metric.labels(**labels).set(value)


@contextmanager
def timer(metric, labels: dict[str, str]) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        observe_histogram(metric, labels, time.perf_counter() - start)
