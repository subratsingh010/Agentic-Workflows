from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from app.domain.models import RetrievedChunk

EVAL_RESOURCE = "goosle_policy_eval.jsonl"


class EvalThresholdError(AssertionError):
    def __init__(self, failures: dict[str, tuple[float, float]]) -> None:
        self.failures = failures
        message = "; ".join(
            f"{metric}={actual:.3f} below required {minimum:.3f}"
            for metric, (actual, minimum) in sorted(failures.items())
        )
        super().__init__(message)


@dataclass(frozen=True)
class PolicyEvalCase:
    id: str
    question: str
    expected_document_id: str
    expected_chunk_id: str
    ground_truth: str
    category: str


def _load_cases_from_lines(lines: Iterable[str]) -> list[PolicyEvalCase]:
    cases: list[PolicyEvalCase] = []
    seen: set[str] = set()
    required = {"id", "question", "expected_document_id", "expected_chunk_id", "ground_truth", "category"}
    for line_number, line in enumerate(lines, start=1):
        raw = line.strip()
        if not raw:
            continue
        payload = json.loads(raw)
        missing = required.difference(payload)
        if missing:
            missing_csv = ", ".join(sorted(missing))
            raise ValueError(f"eval case line {line_number} missing fields: {missing_csv}")
        case = PolicyEvalCase(**{field: payload[field] for field in required})
        if case.id in seen:
            raise ValueError(f"duplicate eval case id {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError("policy eval dataset is empty")
    return cases


def load_eval_cases(path: Path | None = None) -> list[PolicyEvalCase]:
    if path is not None:
        with path.open("r", encoding="utf-8") as handle:
            return _load_cases_from_lines(handle)
    with resources.files(__package__).joinpath(EVAL_RESOURCE).open("r", encoding="utf-8") as handle:
        return _load_cases_from_lines(handle)


def reciprocal_rank(expected_chunk_id: str, retrieved_chunk_ids: list[str]) -> float:
    for index, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id == expected_chunk_id:
            return 1.0 / index
    return 0.0


def normalized_tokens(value: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }
    return {
        token.strip(".,!?;:()[]{}\"'").lower()
        for token in value.split()
        if token.strip() and token.strip(".,!?;:()[]{}\"'").lower() not in stopwords
    }


def token_overlap(reference: str, candidate: str) -> float:
    reference_tokens = normalized_tokens(reference)
    candidate_tokens = normalized_tokens(candidate)
    if not reference_tokens:
        return 0.0
    return len(reference_tokens.intersection(candidate_tokens)) / len(reference_tokens)


def keyword_coverage(case: PolicyEvalCase, chunks: list[RetrievedChunk], top_k: int) -> float:
    context = " ".join(chunk.text for chunk in chunks[:top_k])
    return token_overlap(case.ground_truth, context)


def ndcg_at_k(rank: int | None, top_k: int) -> float:
    if rank is None or rank > top_k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def score_answer(case: PolicyEvalCase, answer: str, chunks: list[RetrievedChunk], top_k: int) -> dict[str, float]:
    context = " ".join(chunk.text for chunk in chunks[:top_k])
    correctness = token_overlap(case.ground_truth, answer)
    groundedness = token_overlap(answer, context)
    completeness = token_overlap(case.ground_truth, answer)
    relevance = token_overlap(case.question, answer)
    llm_judge_score = mean([correctness, groundedness, completeness, relevance])
    return {
        "answer_correctness": correctness,
        "answer_groundedness": groundedness,
        "answer_completeness": completeness,
        "answer_relevance": relevance,
        "llm_judge_score": llm_judge_score,
    }


def score_retrieval_case(case: PolicyEvalCase, chunks: list[RetrievedChunk], top_k: int) -> dict[str, Any]:
    retrieved_ids = [chunk.chunk_id for chunk in chunks[:top_k]]
    retrieved_doc_ids = [chunk.document_id for chunk in chunks[:top_k]]
    rank = next(
        (index for index, chunk_id in enumerate(retrieved_ids, start=1) if chunk_id == case.expected_chunk_id),
        None,
    )
    hit_at_k = case.expected_chunk_id in retrieved_ids
    document_hit_at_k = case.expected_document_id in retrieved_doc_ids
    return {
        "id": case.id,
        "category": case.category,
        "question": case.question,
        "expected_chunk_id": case.expected_chunk_id,
        "retrieved_chunk_ids": retrieved_ids,
        "hit_at_1": retrieved_ids[:1] == [case.expected_chunk_id],
        "hit_at_k": hit_at_k,
        f"recall_at_{top_k}": 1.0 if hit_at_k else 0.0,
        f"precision_at_{top_k}": (1.0 / top_k) if hit_at_k else 0.0,
        f"ndcg_at_{top_k}": ndcg_at_k(rank, top_k),
        "document_hit_at_k": document_hit_at_k,
        "keyword_coverage": keyword_coverage(case, chunks, top_k),
        "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
        "rank": rank,
        "top_score": chunks[0].score if chunks else 0.0,
    }


def summarize_scores(case_scores: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    if not case_scores:
        raise ValueError("cannot summarize empty eval results")

    def avg(field: str) -> float:
        values = [item[field] for item in case_scores if field in item]
        if not values:
            return 0.0
        return mean(1.0 if item is True else 0.0 if item is False else float(item) for item in values)

    def category_summary(items: list[dict[str, Any]]) -> dict[str, float | int]:
        def item_avg(field: str) -> float:
            values = [item[field] for item in items if field in item]
            if not values:
                return 0.0
            return mean(1.0 if item is True else 0.0 if item is False else float(item) for item in values)

        return {
            "cases": len(items),
            "hit_at_1": item_avg("hit_at_1"),
            f"hit_at_{top_k}": item_avg("hit_at_k"),
            f"recall_at_{top_k}": item_avg(f"recall_at_{top_k}"),
            f"precision_at_{top_k}": item_avg(f"precision_at_{top_k}"),
            f"ndcg_at_{top_k}": item_avg(f"ndcg_at_{top_k}"),
            "document_hit_at_k": item_avg("document_hit_at_k"),
            "keyword_coverage": item_avg("keyword_coverage"),
            "mrr": mean(float(item["reciprocal_rank"]) for item in items),
            "llm_judge_score": item_avg("llm_judge_score"),
            "answer_correctness": item_avg("answer_correctness"),
            "answer_groundedness": item_avg("answer_groundedness"),
            "ragas_context_precision": item_avg("ragas_context_precision"),
            "ragas_context_recall": item_avg("ragas_context_recall"),
        }

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in case_scores:
        by_category[item["category"]].append(item)

    categories = {category: category_summary(items) for category, items in sorted(by_category.items())}

    return {
        "cases": len(case_scores),
        "top_k": top_k,
        "hit_at_1": avg("hit_at_1"),
        f"hit_at_{top_k}": avg("hit_at_k"),
        f"recall_at_{top_k}": avg(f"recall_at_{top_k}"),
        f"precision_at_{top_k}": avg(f"precision_at_{top_k}"),
        f"ndcg_at_{top_k}": avg(f"ndcg_at_{top_k}"),
        "document_hit_at_k": avg("document_hit_at_k"),
        "keyword_coverage": avg("keyword_coverage"),
        "mrr": mean(float(item["reciprocal_rank"]) for item in case_scores),
        "llm_judge_score": avg("llm_judge_score"),
        "answer_correctness": avg("answer_correctness"),
        "answer_groundedness": avg("answer_groundedness"),
        "answer_completeness": avg("answer_completeness"),
        "answer_relevance": avg("answer_relevance"),
        "ragas_context_precision": avg("ragas_context_precision"),
        "ragas_context_recall": avg("ragas_context_recall"),
        "categories": categories,
    }


def assert_thresholds(summary: dict[str, Any], thresholds: dict[str, float]) -> None:
    failures: dict[str, tuple[float, float]] = {}
    for metric, minimum in thresholds.items():
        actual = float(summary.get(metric, 0.0))
        if actual < minimum:
            failures[metric] = (actual, minimum)
    if failures:
        raise EvalThresholdError(failures)
