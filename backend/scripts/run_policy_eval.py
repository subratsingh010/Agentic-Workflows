from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import httpx  # noqa: E402

from app.adapters.rag.hybrid import InMemoryHybridRetriever  # noqa: E402
from app.application.evaluation.policy_eval import (  # noqa: E402
    assert_thresholds,
    load_eval_cases,
    score_answer,
    score_retrieval_case,
    summarize_scores,
    token_overlap,
)
from app.domain.models import ActorContext, RetrievedChunk  # noqa: E402


def _default_dataset_path() -> Path | None:
    candidate = ROOT / "data/evals/goosle_policy_eval.jsonl"
    return candidate if candidate.exists() else None


async def _run_retriever_eval(
    top_k: int, retrieval_mode: str = "hybrid", fusion_strategy: str = "weighted"
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases = load_eval_cases(_default_dataset_path())
    retriever = InMemoryHybridRetriever(
        retrieval_mode=retrieval_mode,
        fusion_strategy=fusion_strategy,
    )
    actor = ActorContext(
        subject="eval-user",
        employee_id="EVAL-001",
        email="eval.user@goosle.example",
        department="engineering",
        country="US",
        roles={"employee"},
    )
    scores: list[dict[str, Any]] = []
    for case in cases:
        chunks = await retriever.retrieve(case.question, actor, top_k=top_k)
        scores.append(score_retrieval_case(case, chunks, top_k))
    return scores, summarize_scores(scores, top_k)


async def _compare_retrieval_modes(top_k: int) -> dict[str, dict[str, Any]]:
    comparisons: dict[str, dict[str, Any]] = {}
    for name, retrieval_mode, fusion_strategy in [
        ("dense", "dense", "weighted"),
        ("sparse_bm25", "sparse", "weighted"),
        ("hybrid_weighted", "hybrid", "weighted"),
        ("hybrid_rrf", "hybrid", "rrf"),
    ]:
        _, summary = await _run_retriever_eval(top_k, retrieval_mode, fusion_strategy)
        comparisons[name] = summary
    return comparisons


async def _run_live_api_eval(
    base_url: str, bearer_token: str, top_k: int, timeout: float
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    cases = load_eval_cases(_default_dataset_path())
    scores: list[dict[str, Any]] = []
    ragas_records: list[dict[str, Any]] = []
    async with httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=timeout) as client:
        for case in cases:
            response = await client.post(
                "/chat",
                headers={"Authorization": f"Bearer {bearer_token}"},
                json={"message": case.question, "metadata": {"eval_case_id": case.id}},
            )
            response.raise_for_status()
            payload = response.json()
            chunks = [
                RetrievedChunk(
                    document_id=citation["document_id"],
                    title=citation["title"],
                    chunk_id=citation["chunk_id"],
                    text=citation.get("excerpt", ""),
                    score=float(citation.get("score", 0.0)),
                )
                for citation in payload.get("citations", [])
            ]
            item = score_retrieval_case(case, chunks, top_k)
            answer = payload.get("answer", "")
            item.update(score_answer(case, answer, chunks, top_k))
            item["answer_overlap"] = token_overlap(case.ground_truth, answer)
            item["intent"] = payload.get("intent")
            scores.append(item)
            ragas_records.append(
                {
                    "user_input": case.question,
                    "response": answer,
                    "retrieved_contexts": [chunk.text for chunk in chunks],
                    "reference": case.ground_truth,
                    "expected_chunk_id": case.expected_chunk_id,
                    "category": case.category,
                }
            )
    summary = summarize_scores(scores, top_k)
    summary["mean_answer_overlap"] = sum(
        float(item.get("answer_overlap", 0.0)) for item in scores
    ) / len(scores)
    return scores, summary, ragas_records


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run Goosle policy RAG evaluation")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--retrieval-mode", choices=["dense", "sparse", "hybrid"], default="hybrid")
    parser.add_argument("--fusion-strategy", choices=["weighted", "rrf"], default="weighted")
    parser.add_argument("--compare-retrieval-modes", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/evals/policy_eval_report.json",
    )
    parser.add_argument(
        "--live-api",
        default="",
        help="Optional running API base URL, for example http://localhost:8000",
    )
    parser.add_argument("--bearer-token", default="dev-token")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--min-hit-at-1",
        type=float,
        default=None,
        help="Optional gate: fail if hit_at_1 is below this value",
    )
    parser.add_argument(
        "--min-hit-at-k",
        type=float,
        default=None,
        help="Optional gate: fail if hit_at_k is below this value",
    )
    parser.add_argument(
        "--min-document-hit-at-k",
        type=float,
        default=None,
        help="Optional gate: fail if document_hit_at_k is below this value",
    )
    parser.add_argument(
        "--min-mrr",
        type=float,
        default=None,
        help="Optional gate: fail if mrr is below this value",
    )
    parser.add_argument(
        "--min-precision-at-k",
        type=float,
        default=None,
        help="Optional gate: fail if precision_at_k is below this value",
    )
    parser.add_argument(
        "--min-ndcg-at-k",
        type=float,
        default=None,
        help="Optional gate: fail if ndcg_at_k is below this value",
    )
    parser.add_argument(
        "--min-keyword-coverage",
        type=float,
        default=None,
        help="Optional gate: fail if keyword coverage is below this value",
    )
    parser.add_argument(
        "--min-llm-judge-score",
        type=float,
        default=None,
        help="Optional gate for live API answer quality",
    )
    parser.add_argument(
        "--ragas-output",
        type=Path,
        default=ROOT / "outputs/evals/ragas_policy_eval.jsonl",
        help="Ragas-style JSONL written when --live-api is used",
    )
    args = parser.parse_args()

    if args.live_api:
        case_scores, summary, ragas_records = await _run_live_api_eval(
            args.live_api, args.bearer_token, args.top_k, args.timeout
        )
        _write_jsonl(args.ragas_output, ragas_records)
        summary["ragas_output"] = str(args.ragas_output)
        mode = "live_api"
    else:
        case_scores, summary = await _run_retriever_eval(
            args.top_k, args.retrieval_mode, args.fusion_strategy
        )
        if args.compare_retrieval_modes:
            summary["mode_comparison"] = await _compare_retrieval_modes(args.top_k)
        mode = f"retriever:{args.retrieval_mode}:{args.fusion_strategy}"

    hit_at_k_metric = f"hit_at_{args.top_k}"
    thresholds = {
        "hit_at_1": args.min_hit_at_1,
        hit_at_k_metric: args.min_hit_at_k,
        f"precision_at_{args.top_k}": args.min_precision_at_k,
        f"ndcg_at_{args.top_k}": args.min_ndcg_at_k,
        "document_hit_at_k": args.min_document_hit_at_k,
        "keyword_coverage": args.min_keyword_coverage,
        "mrr": args.min_mrr,
        "llm_judge_score": args.min_llm_judge_score,
    }
    enabled_thresholds = {key: value for key, value in thresholds.items() if value is not None}
    if enabled_thresholds:
        assert_thresholds(summary, enabled_thresholds)

    report = {"mode": mode, "summary": summary, "cases": case_scores}
    _write_report(args.output, report)
    print(
        json.dumps(
            {"mode": mode, "output": str(args.output), "summary": summary},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
