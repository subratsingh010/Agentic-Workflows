from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from app.domain.models import RetrievedChunk


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 120
    chunk_overlap: int = 20

    def __post_init__(self) -> None:
        if self.chunk_size < 20:
            raise ValueError("chunk_size must be at least 20 words")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "document"


def _words(text: str) -> list[str]:
    return [word for word in text.split() if word.strip()]


def _windows(words: list[str], config: ChunkingConfig) -> Iterable[list[str]]:
    step = config.chunk_size - config.chunk_overlap
    for start in range(0, len(words), step):
        window = words[start : start + config.chunk_size]
        if window:
            yield window
        if start + config.chunk_size >= len(words):
            break


def _metadata_from_path(path: Path, document_id: str) -> dict[str, Any]:
    topic = path.stem.removeprefix("goosle-").replace("-", " ")
    return {
        "company": "Goosle",
        "countries": ["global"],
        "departments": ["all"],
        "topic": topic,
        "source_path": str(path),
        "document_id": document_id,
    }


def _chunk_markdown_block(raw: str, path: Path, fallback_title: str, config: ChunkingConfig) -> list[RetrievedChunk]:
    title_match = re.search(r"^##?\s+(.+)$", raw, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else fallback_title
    document_id_match = re.search(r"<!--\s*document_id:\s*([^>]+?)\s*-->", raw)
    document_id = document_id_match.group(1).strip() if document_id_match else _slug(title)

    sections = re.split(r"(?=^###\s+)", raw, flags=re.MULTILINE)
    chunks: list[RetrievedChunk] = []
    metadata = _metadata_from_path(path, document_id) | {"topic": title.lower().replace("goosle ", "")}
    semantic_index = 1

    for section in sections:
        if not section.strip() or section.lstrip().startswith("# ") or section.lstrip().startswith("## "):
            continue
        chunk_id_match = re.search(r"<!--\s*chunk_id:\s*([^>]+?)\s*-->", section)
        base_chunk_id = chunk_id_match.group(1).strip() if chunk_id_match else f"{_slug(document_id)}-{semantic_index:04d}"
        cleaned = re.sub(r"<!--.*?-->", " ", section, flags=re.DOTALL)
        cleaned = re.sub(r"^#+\s+.*$", " ", cleaned, flags=re.MULTILINE)
        section_words = _words(cleaned)
        if not section_words:
            continue
        windows = list(_windows(section_words, config))
        for split_index, window in enumerate(windows, start=1):
            chunk_id = base_chunk_id if len(windows) == 1 else f"{base_chunk_id}-part-{split_index}"
            chunks.append(
                RetrievedChunk(
                    document_id=document_id,
                    title=title,
                    chunk_id=chunk_id,
                    text=" ".join(window),
                    metadata=metadata | {"chunk_size": config.chunk_size, "chunk_overlap": config.chunk_overlap},
                )
            )
        semantic_index += 1

    if chunks:
        return chunks

    all_words = _words(re.sub(r"^#+\s+.*$", " ", raw, flags=re.MULTILINE))
    return [
        RetrievedChunk(
            document_id=document_id,
            title=title,
            chunk_id=f"{_slug(document_id)}-{index:04d}",
            text=" ".join(window),
            metadata=metadata | {"chunk_size": config.chunk_size, "chunk_overlap": config.chunk_overlap},
        )
        for index, window in enumerate(_windows(all_words, config), start=1)
    ]


def chunk_markdown_document(path: Path, config: ChunkingConfig) -> list[RetrievedChunk]:
    raw = path.read_text(encoding="utf-8")
    fallback_title_match = re.search(r"^#\s+(.+)$", raw, flags=re.MULTILINE)
    fallback_title = fallback_title_match.group(1).strip() if fallback_title_match else path.stem.replace("-", " ").title()

    blocks = re.split(r"(?=^##\s+)", raw, flags=re.MULTILINE)
    chunks: list[RetrievedChunk] = []
    for block in blocks:
        if not block.strip() or block.lstrip().startswith("# "):
            continue
        chunks.extend(_chunk_markdown_block(block, path, fallback_title, config))

    if chunks:
        return chunks
    return _chunk_markdown_block(raw, path, fallback_title, config)


def build_chunks_from_source_dir(source_dir: Path, config: ChunkingConfig) -> list[RetrievedChunk]:
    chunks: list[RetrievedChunk] = []
    for path in sorted(source_dir.glob("*.md")):
        chunks.extend(chunk_markdown_document(path, config))
    if not chunks:
        raise ValueError(f"no markdown policy documents found in {source_dir}")
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("generated duplicate chunk ids")
    return chunks


def write_chunks_jsonl(chunks: list[RetrievedChunk], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.model_dump(), sort_keys=True) + "\n")
