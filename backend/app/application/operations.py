from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.adapters.rag import hybrid as hybrid_module
from app.adapters.rag import policy_seed
from app.adapters.rag.embeddings import get_embedding_model
from app.adapters.rag.hybrid import (
    HeuristicCrossEncoderReranker,
    InMemoryHybridRetriever,
    MilvusHybridRetriever,
    NativeMilvusHybridRetriever,
    POLICY_CHUNKS,
    ScoreReranker,
)
from app.adapters.rag.llm import MockLLMGenerator, OllamaLLMGenerator, OpenAILLMGenerator
from app.application.evaluation.policy_eval import (
    load_eval_cases,
    score_answer,
    score_retrieval_case,
    summarize_scores,
)
from app.application.ingestion import ChunkingConfig, build_chunks_from_source_dir, write_chunks_jsonl
from app.core.config import Settings
from app.domain.models import ActorContext


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def eval_report_path() -> Path:
    return project_root() / "outputs/evals/policy_eval_report.json"


def packaged_chunks_path() -> Path:
    return project_root() / "backend/app/adapters/rag/goosle_policy_chunks.jsonl"


def source_chunks_path() -> Path:
    return project_root() / "data/policies/goosle_policy_chunks.jsonl"


def settings_with_overrides(settings: Settings, overrides: dict[str, Any] | None = None) -> Settings:
    if not overrides:
        return settings
    allowed = {
        "rag_chunk_size",
        "rag_chunk_overlap",
        "rag_retrieval_mode",
        "rag_fusion_strategy",
        "rag_reranker_provider",
        "rag_top_k",
        "rag_dense_weight",
        "rag_sparse_weight",
        "rag_candidate_multiplier",
        "milvus_native_hybrid",
        "milvus_vector_dim",
        "embedding_device",
        "embedding_model",
        "embedding_provider",
    }
    clean = {key: value for key, value in overrides.items() if key in allowed and value is not None}
    return settings.model_copy(update=clean)




def _build_embedding_model(settings):
    return get_embedding_model(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )


def _build_reranker(settings: Settings):
    if not settings.rag_rerank_enabled:
        return ScoreReranker()
    provider = settings.rag_reranker_provider.lower()
    if provider in {"heuristic", "cross_encoder", "heuristic_cross_encoder"}:
        return HeuristicCrossEncoderReranker()
    return ScoreReranker()


def _build_llm(settings: Settings):
    provider = settings.llm_provider.lower()
    if provider == "openai":
        return OpenAILLMGenerator(
            model=settings.openai_model,
            timeout_seconds=settings.openai_timeout_seconds,
        )
    if provider == "ollama":
        return OllamaLLMGenerator(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout_seconds=settings.ollama_timeout_seconds,
        )
    return MockLLMGenerator()


def _build_retriever(settings: Settings):
    common = {
        "dense_weight": settings.rag_dense_weight,
        "sparse_weight": settings.rag_sparse_weight,
        "retrieval_mode": settings.rag_retrieval_mode.lower(),
        "fusion_strategy": settings.rag_fusion_strategy.lower(),
        "candidate_multiplier": settings.rag_candidate_multiplier,
        "rrf_k": settings.rag_rrf_k,
        "bm25_k1": settings.rag_bm25_k1,
        "bm25_b": settings.rag_bm25_b,
        "embedding_model": _build_embedding_model(settings),
    }
    if settings.rag_backend.lower() == "milvus":
        retriever_cls = NativeMilvusHybridRetriever if settings.milvus_native_hybrid else MilvusHybridRetriever
        return retriever_cls(
            host=settings.milvus_host,
            port=settings.milvus_port,
            collection_name=settings.milvus_collection,
            vector_dim=settings.milvus_vector_dim,
            **common,
        )
    return InMemoryHybridRetriever(**common)


def _latest_eval_summary() -> dict[str, Any] | None:
    path = eval_report_path()
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload.get("summary")


def knowledge_status(settings: Settings) -> dict[str, Any]:
    cases = load_eval_cases()
    return {
        "rag_backend": settings.rag_backend,
        "corpus_chunks": len(POLICY_CHUNKS),
        "eval_cases": len(cases),
        "collection": settings.milvus_collection
        if settings.rag_backend.lower() == "milvus"
        else "in_memory",
        "latest_eval": _latest_eval_summary(),
        "chunk_size": settings.rag_chunk_size,
        "chunk_overlap": settings.rag_chunk_overlap,
        "source_dir": settings.policy_source_dir,
        "milvus_native_hybrid": settings.milvus_native_hybrid,
        "retrieval_mode": settings.rag_retrieval_mode.lower(),
        "fusion_strategy": settings.rag_fusion_strategy.lower(),
        "reranker_provider": settings.rag_reranker_provider.lower(),
        "candidate_multiplier": settings.rag_candidate_multiplier,
        "dense_weight": settings.rag_dense_weight,
        "sparse_weight": settings.rag_sparse_weight,
        "top_k": settings.rag_top_k,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "embedding_device": settings.embedding_device,
        "embedding_dim": settings.milvus_vector_dim,
    }


def rebuild_seed_corpus(settings: Settings) -> dict[str, Any]:
    config = ChunkingConfig(
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )
    chunks = build_chunks_from_source_dir(project_root() / settings.policy_source_dir, config)
    write_chunks_jsonl(chunks, source_chunks_path())
    write_chunks_jsonl(chunks, packaged_chunks_path())
    POLICY_CHUNKS[:] = chunks
    policy_seed.POLICY_CHUNKS[:] = chunks
    hybrid_module.POLICY_CHUNKS[:] = chunks
    policy_seed.load_policy_chunks.cache_clear()
    return {
        "status": "rebuilt",
        "chunks": len(chunks),
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "source": settings.policy_source_dir,
    }


def ingest_seed_corpus(settings: Settings) -> dict[str, Any]:
    retriever = _build_retriever(settings)
    if isinstance(retriever, (MilvusHybridRetriever, NativeMilvusHybridRetriever)):
        indexed_chunks = retriever.rebuild_seed_collection()
        status = "indexed_in_milvus"
    else:
        indexed_chunks = len(POLICY_CHUNKS)
        status = "indexed_in_memory"
    return {
        "status": status,
        "rag_backend": settings.rag_backend,
        "indexed_chunks": indexed_chunks,
        "source": "packaged_goosle_policy_seed",
    }


async def run_policy_eval(settings: Settings, top_k: int | None = None) -> dict[str, Any]:
    effective_top_k = top_k or settings.rag_top_k
    retriever = _build_retriever(settings)
    reranker = _build_reranker(settings)
    llm = _build_llm(settings)
    cases = load_eval_cases()
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
        retrieved_chunks = await retriever.retrieve(case.question, actor, top_k=effective_top_k)
        reranked_chunks = await reranker.rerank(case.question, retrieved_chunks)
        answer_chunks = reranked_chunks[:effective_top_k]
        generated_answer = await llm.answer_policy(case.question, answer_chunks)
        item = score_retrieval_case(case, answer_chunks, top_k=effective_top_k)
        item.update(score_answer(case, generated_answer, answer_chunks, top_k=effective_top_k))
        item.update(
            {
                "generated_answer": generated_answer,
                "ground_truth": case.ground_truth,
                "pre_rerank_chunk_ids": [
                    chunk.chunk_id for chunk in retrieved_chunks[:effective_top_k]
                ],
                "post_rerank_chunk_ids": [chunk.chunk_id for chunk in answer_chunks],
                "retrieval_backend": (
                    answer_chunks[0].metadata.get("retrieval_backend", settings.rag_backend)
                    if answer_chunks
                    else settings.rag_backend
                ),
                "reranker": (
                    settings.rag_reranker_provider.lower()
                    if settings.rag_rerank_enabled
                    else "score"
                ),
            }
        )
        scores.append(item)

    summary = summarize_scores(scores, effective_top_k)
    pipeline = {
        "embedding": {
            "provider": settings.embedding_provider,
            "model": settings.embedding_model,
            "device": settings.embedding_device,
            "dim": settings.milvus_vector_dim,
        },
        "ingestion": {
            "source_dir": settings.policy_source_dir,
            "chunk_size": settings.rag_chunk_size,
            "chunk_overlap": settings.rag_chunk_overlap,
            "chunks": len(POLICY_CHUNKS),
        },
        "retrieval": {
            "backend": settings.rag_backend,
            "native_milvus_hybrid": settings.milvus_native_hybrid,
            "collection": (
                settings.milvus_collection
                if settings.rag_backend.lower() == "milvus"
                else "in_memory"
            ),
            "mode": settings.rag_retrieval_mode.lower(),
            "fusion": settings.rag_fusion_strategy.lower(),
            "dense_weight": settings.rag_dense_weight,
            "sparse_weight": settings.rag_sparse_weight,
            "candidate_multiplier": settings.rag_candidate_multiplier,
        },
        "reranker": {
            "enabled": settings.rag_rerank_enabled,
            "provider": settings.rag_reranker_provider.lower(),
        },
        "llm": {
            "provider": settings.llm_provider,
            "model": (
                settings.openai_model
                if settings.llm_provider.lower() == "openai"
                else settings.ollama_model
            ),
        },
    }
    report = {
        "mode": settings.rag_backend,
        "retrieval_mode": settings.rag_retrieval_mode.lower(),
        "fusion_strategy": settings.rag_fusion_strategy.lower(),
        "pipeline": pipeline,
        "summary": summary,
        "cases": scores,
    }
    path = eval_report_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"summary": summary, "pipeline": pipeline, "cases": scores[:5], "output": str(path)}
