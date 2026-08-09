from __future__ import annotations

import pytest

from app.adapters.rag.hybrid import HeuristicCrossEncoderReranker, InMemoryHybridRetriever, POLICY_CHUNKS
from app.application.evaluation.policy_eval import (
    EvalThresholdError,
    assert_thresholds,
    keyword_coverage,
    load_eval_cases,
    ndcg_at_k,
    reciprocal_rank,
    score_answer,
    score_retrieval_case,
    summarize_scores,
    token_overlap,
)
from app.application.ingestion import ChunkingConfig, chunk_markdown_document
from app.domain.models import ActorContext


def test_policy_seed_dataset_is_large_and_unique():
    chunk_ids = [chunk.chunk_id for chunk in POLICY_CHUNKS]

    assert len(POLICY_CHUNKS) >= 60
    assert len(chunk_ids) == len(set(chunk_ids))
    assert {chunk.metadata.get("company") for chunk in POLICY_CHUNKS} == {"Goosle"}


@pytest.mark.eval
def test_policy_eval_dataset_shape_and_coverage():
    cases = load_eval_cases()
    chunk_ids = {chunk.chunk_id for chunk in POLICY_CHUNKS}
    categories = {case.category for case in cases}

    assert len(cases) >= 60
    assert {case.expected_chunk_id for case in cases}.issubset(chunk_ids)
    assert {"leave-policy", "jira-time-policy", "security-policy", "remote-work", "expense-policy"}.issubset(categories)


def test_policy_eval_metric_helpers():
    assert reciprocal_rank("b", ["a", "b", "c"]) == 0.5
    assert reciprocal_rank("x", ["a", "b", "c"]) == 0.0
    assert token_overlap("vacation leave approval", "manager approval for vacation leave") == 1.0


@pytest.mark.asyncio
@pytest.mark.eval
async def test_retriever_eval_pipeline_produces_summary():
    cases = load_eval_cases()[:12]
    retriever = InMemoryHybridRetriever()
    actor = ActorContext(subject="eval", employee_id="EVAL-001", roles={"employee"})

    scores = []
    for case in cases:
        chunks = await retriever.retrieve(case.question, actor, top_k=5)
        scores.append(score_retrieval_case(case, chunks, top_k=5))

    summary = summarize_scores(scores, top_k=5)

    assert summary["cases"] == 12
    assert 0.0 <= summary["hit_at_1"] <= 1.0
    assert 0.0 <= summary["hit_at_5"] <= 1.0
    assert 0.0 <= summary["mrr"] <= 1.0


def test_policy_eval_threshold_gate():
    summary = {"hit_at_1": 0.8, "hit_at_5": 0.9, "document_hit_at_k": 1.0, "mrr": 0.85}

    assert_thresholds(summary, {"hit_at_1": 0.7, "hit_at_5": 0.9})
    with pytest.raises(EvalThresholdError):
        assert_thresholds(summary, {"mrr": 0.95})


def test_markdown_chunker_respects_size_and_overlap(tmp_path):
    source = tmp_path / "sample-policy.md"
    words = " ".join(f"word{i}" for i in range(95))
    source.write_text(
        "# Sample Policy\n<!-- document_id: sample-policy -->\n\n"
        "### sample-001\n<!-- chunk_id: sample-001 -->\n"
        f"{words}\n",
        encoding="utf-8",
    )

    chunks = chunk_markdown_document(source, ChunkingConfig(chunk_size=40, chunk_overlap=10))

    assert [chunk.chunk_id for chunk in chunks] == [
        "sample-001-part-1",
        "sample-001-part-2",
        "sample-001-part-3",
    ]
    assert all(len(chunk.text.split()) <= 40 for chunk in chunks)
    assert chunks[0].text.split()[-10:] == chunks[1].text.split()[:10]


def test_policy_eval_retrieval_metrics():
    assert ndcg_at_k(1, 5) == 1.0
    assert round(ndcg_at_k(2, 5), 3) == 0.631
    assert ndcg_at_k(None, 5) == 0.0


def test_policy_eval_answer_and_keyword_metrics():
    cases = load_eval_cases()
    case = cases[0]
    chunk = next(chunk for chunk in POLICY_CHUNKS if chunk.chunk_id == case.expected_chunk_id)

    assert keyword_coverage(case, [chunk], top_k=1) > 0.5
    answer_scores = score_answer(case, case.ground_truth, [chunk], top_k=1)
    assert answer_scores["answer_correctness"] == 1.0
    assert 0.0 <= answer_scores["answer_groundedness"] <= 1.0
    assert 0.0 <= answer_scores["llm_judge_score"] <= 1.0


@pytest.mark.asyncio
async def test_retrieval_modes_return_ranked_chunks():
    actor = ActorContext(subject="eval", employee_id="EVAL-001", roles={"employee"})
    for retrieval_mode, fusion_strategy in [
        ("dense", "weighted"),
        ("sparse", "weighted"),
        ("hybrid", "weighted"),
        ("hybrid", "rrf"),
    ]:
        retriever = InMemoryHybridRetriever(
            retrieval_mode=retrieval_mode,
            fusion_strategy=fusion_strategy,
            candidate_multiplier=1,
        )
        chunks = await retriever.retrieve("What is the leave policy?", actor, top_k=5)
        assert len(chunks) == 5
        assert chunks == sorted(chunks, key=lambda item: item.score, reverse=True)


@pytest.mark.asyncio
async def test_heuristic_cross_encoder_reranker_updates_scores():
    retriever = InMemoryHybridRetriever(candidate_multiplier=1)
    actor = ActorContext(subject="eval", employee_id="EVAL-001", roles={"employee"})
    chunks = await retriever.retrieve("How do I request vacation leave?", actor, top_k=5)
    reranked = await HeuristicCrossEncoderReranker().rerank("How do I request vacation leave?", chunks)

    assert len(reranked) == 5
    assert all("reranker" in chunk.metadata for chunk in reranked)
    assert reranked == sorted(reranked, key=lambda item: item.score, reverse=True)
