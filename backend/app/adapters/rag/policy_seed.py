from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Iterable

from app.domain.models import RetrievedChunk

SEED_RESOURCE = "goosle_policy_chunks.jsonl"


def _parse_chunks(lines: Iterable[str]) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    seen: set[str] = set()
    for line_number, line in enumerate(lines, start=1):
        raw = line.strip()
        if not raw:
            continue
        payload = json.loads(raw)
        chunk = RetrievedChunk(**payload)
        if chunk.chunk_id in seen:
            raise ValueError(f"duplicate policy chunk_id {chunk.chunk_id!r}")
        seen.add(chunk.chunk_id)
        chunks.append(chunk)
    if not chunks:
        raise ValueError("policy seed dataset is empty")
    return chunks


@lru_cache(maxsize=1)
def load_policy_chunks() -> tuple[RetrievedChunk, ...]:
    with resources.files(__package__).joinpath(SEED_RESOURCE).open("r", encoding="utf-8") as handle:
        return tuple(_parse_chunks(handle))


def load_policy_chunks_from_path(path: Path) -> list[RetrievedChunk]:
    with path.open("r", encoding="utf-8") as handle:
        return _parse_chunks(handle)


POLICY_CHUNKS = list(load_policy_chunks())
