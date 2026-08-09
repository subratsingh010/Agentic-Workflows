from __future__ import annotations

from functools import lru_cache
from typing import Protocol


class EmbeddingModel(Protocol):
    @property
    def dim(self) -> int: ...

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbeddingModel:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str = "cpu",
        normalize_embeddings: bool = True,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for embeddings"
            ) from exc
        self._model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        self._normalize_embeddings = normalize_embeddings
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize_embeddings,
            show_progress_bar=False,
        )
        return [vector.astype(float).tolist() for vector in vectors]


@lru_cache(maxsize=4)
def get_embedding_model(
    model_name: str = "BAAI/bge-small-en-v1.5",
    device: str = "cpu",
) -> EmbeddingModel:
    return SentenceTransformerEmbeddingModel(model_name=model_name, device=device)
