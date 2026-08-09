from __future__ import annotations

from dataclasses import dataclass

from app.domain.models import Citation, CitationEvidence, GroundingReport, RetrievedChunk

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "for",
    "from",
    "i",
    "if",
    "in",
    "is",
    "it",
    "may",
    "of",
    "on",
    "or",
    "should",
    "the",
    "to",
    "with",
    "you",
    "your",
}


@dataclass(frozen=True)
class FaithfulnessPolicy:
    min_score: float = 0.35
    min_citations: int = 1


def _terms(value: str) -> set[str]:
    terms: set[str] = set()
    for raw in value.split():
        token = raw.strip(".,!?;:()[]{}\"'").lower()
        if len(token) < 3 or token in _STOPWORDS:
            continue
        terms.add(token)
    return terms


def _chunk_terms(chunks: list[RetrievedChunk]) -> dict[str, set[str]]:
    return {chunk.chunk_id: _terms(f"{chunk.title} {chunk.text}") for chunk in chunks}


def assess_grounding(answer: str, chunks: list[RetrievedChunk], policy: FaithfulnessPolicy) -> GroundingReport:
    if not chunks:
        return GroundingReport(
            grounded=False,
            faithfulness_score=0,
            citation_count=0,
            unsupported_terms=sorted(_terms(answer)),
            guardrail_action="blocked",
        )
    answer_terms = _terms(answer)
    context_terms = set().union(*_chunk_terms(chunks).values()) if chunks else set()
    if not answer_terms:
        return GroundingReport(
            grounded=True,
            faithfulness_score=1,
            citation_count=len(chunks),
            guardrail_action="pass",
        )
    supported = answer_terms.intersection(context_terms)
    unsupported = answer_terms.difference(context_terms)
    score = len(supported) / len(answer_terms)
    grounded = score >= policy.min_score and len(chunks) >= policy.min_citations
    return GroundingReport(
        grounded=grounded,
        faithfulness_score=round(score, 4),
        citation_count=len(chunks),
        supported_terms=sorted(supported),
        unsupported_terms=sorted(unsupported),
        guardrail_action="pass" if grounded else "blocked",
    )


def build_citations(chunks: list[RetrievedChunk], answer: str) -> list[Citation]:
    answer_terms = _terms(answer)
    terms_by_chunk = _chunk_terms(chunks)
    citations: list[Citation] = []
    for chunk in chunks:
        matched = sorted(answer_terms.intersection(terms_by_chunk.get(chunk.chunk_id, set())))
        support_score = len(matched) / len(answer_terms) if answer_terms else 1.0
        citations.append(
            Citation(
                document_id=chunk.document_id,
                title=chunk.title,
                chunk_id=chunk.chunk_id,
                score=chunk.score,
                excerpt=chunk.text[:240],
                source_path=chunk.metadata.get("source_path"),
                dense_score=chunk.dense_score,
                sparse_score=chunk.sparse_score,
                evidence=CitationEvidence(matched_terms=matched[:20], support_score=round(support_score, 4)),
            )
        )
    return citations


def blocked_answer() -> str:
    return (
        "I do not have enough grounded policy context to answer that safely. "
        "Please rephrase the question or ask about a specific policy area."
    )
