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


class FastEmbedEmbeddingModel:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError("fastembed is required for FastEmbed embeddings") from exc
        self._model_name = model_name
        self._model = TextEmbedding(model_name=model_name)
        self._dim = len(next(self._model.embed(["dimension probe"])))

    @property
    def dim(self) -> int:
        return self._dim

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vector.astype(float).tolist() for vector in self._model.embed(texts)]


@lru_cache(maxsize=4)
def get_embedding_model(
    model_name: str = "BAAI/bge-small-en-v1.5",
    device: str = "cpu",
    provider: str = "sentence_transformers",
) -> EmbeddingModel:
    normalized = provider.lower()
    if normalized == "fastembed":
        return FastEmbedEmbeddingModel(model_name=model_name)
    if normalized == "sentence_transformers":
        return SentenceTransformerEmbeddingModel(model_name=model_name, device=device)
    raise ValueError(f"unsupported embedding provider: {provider}")
