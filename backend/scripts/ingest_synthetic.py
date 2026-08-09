from app.adapters.rag.hybrid import SYNTHETIC_POLICY_CHUNKS


def main() -> None:
    for chunk in SYNTHETIC_POLICY_CHUNKS:
        print(f"would index {chunk.document_id}/{chunk.chunk_id}: {chunk.title}")


if __name__ == "__main__":
    main()

