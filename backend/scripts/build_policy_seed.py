from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.application.ingestion import (  # noqa: E402
    ChunkingConfig,
    build_chunks_from_source_dir,
    write_chunks_jsonl,
)
from app.core.config import get_settings  # noqa: E402


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build policy seed chunks from markdown sources")
    parser.add_argument("--source-dir", type=Path, default=PROJECT_ROOT / settings.policy_source_dir)
    parser.add_argument("--chunk-size", type=int, default=settings.rag_chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=settings.rag_chunk_overlap)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/policies/goosle_policy_chunks.jsonl")
    parser.add_argument(
        "--package-output",
        type=Path,
        default=BACKEND_ROOT / "app/adapters/rag/goosle_policy_chunks.jsonl",
    )
    args = parser.parse_args()

    config = ChunkingConfig(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    chunks = build_chunks_from_source_dir(args.source_dir, config)
    write_chunks_jsonl(chunks, args.output)
    args.package_output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.output, args.package_output)
    print(
        f"built {len(chunks)} chunks from {args.source_dir} "
        f"with chunk_size={config.chunk_size}, overlap={config.chunk_overlap}"
    )


if __name__ == "__main__":
    main()
