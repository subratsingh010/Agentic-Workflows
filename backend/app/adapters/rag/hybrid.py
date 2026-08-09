import math
import time
from collections import Counter
from functools import cached_property
from typing import Literal

from app.adapters.rag.embeddings import EmbeddingModel
from app.adapters.rag.policy_seed import POLICY_CHUNKS
from app.adapters.observability.metrics import (
    RAG_ERRORS,
    RAG_RERANK_SECONDS,
    RAG_RETRIEVAL_SECONDS,
    RAG_RETRIEVED_CHUNKS,
    increment,
    observe_histogram,
)
from app.application.ports import PolicyRetriever, Reranker
from app.domain.models import ActorContext, RetrievedChunk

SYNTHETIC_POLICY_CHUNKS = POLICY_CHUNKS
RetrievalMode = Literal["dense", "sparse", "hybrid"]
FusionStrategy = Literal["weighted", "rrf"]


def _tokens(text: str) -> set[str]:
    return set(_token_list(text))


def _token_list(text: str) -> list[str]:
    return [
        token.strip(".,!?;:()[]{}\"'").lower()
        for token in text.split()
        if token.strip(".,!?;:()[]{}\"'")
    ]



def _first_specific(values: list[str], generic: str) -> str:
    for value in values:
        if value and value != generic:
            return value
    return ""


def _access_fields(chunk: RetrievedChunk) -> dict[str, object]:
    countries = [str(value) for value in chunk.metadata.get("countries", [])]
    departments = [str(value) for value in chunk.metadata.get("departments", [])]
    return {
        "country_global": not countries or "global" in countries,
        "country_code": _first_specific(countries, "global"),
        "department_all": not departments or "all" in departments,
        "department_code": _first_specific(departments, "all"),
    }


def _milvus_literal(value: str | None) -> str:
    return (value or "").replace("\\", "\\\\").replace('"', '\\"')


def _milvus_access_expr(actor: ActorContext) -> str:
    country = _milvus_literal(actor.country)
    department = _milvus_literal(actor.department)
    return (
        f'(country_global == true or country_code == "{country}") '
        f'and (department_all == true or department_code == "{department}")'
    )


def _actor_can_access_chunk(actor: ActorContext, chunk: RetrievedChunk) -> bool:
    countries = set(chunk.metadata.get("countries", []))
    departments = set(chunk.metadata.get("departments", []))
    country_ok = not countries or actor.country in countries or "global" in countries
    department_ok = not departments or actor.department in departments or "all" in departments
    return country_ok and department_ok


def cosine_score(left: list[float], right: list[float]) -> float:
    score = sum(a * b for a, b in zip(left, right, strict=False))
    return max((score + 1.0) / 2.0, 0.0)


def lexical_score(query: str, text: str) -> float:
    query_terms = _tokens(query)
    text_terms = _tokens(text)
    if not query_terms:
        return 0.0
    return len(query_terms.intersection(text_terms)) / max(len(query_terms), 1)


class BM25SparseScorer:
    def __init__(self, chunks: list[RetrievedChunk], k1: float = 1.5, b: float = 0.75) -> None:
        self._chunks = chunks
        self._k1 = k1
        self._b = b
        self._doc_tokens = [_token_list(chunk.text) for chunk in chunks]
        self._doc_lengths = [len(tokens) for tokens in self._doc_tokens]
        self._avg_doc_length = sum(self._doc_lengths) / max(len(self._doc_lengths), 1)
        doc_frequency: Counter[str] = Counter()
        for tokens in self._doc_tokens:
            doc_frequency.update(set(tokens))
        document_count = len(chunks)
        self._idf = {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in doc_frequency.items()
        }

    def scores(self, query: str) -> dict[str, float]:
        query_terms = _token_list(query)
        raw_scores: dict[str, float] = {}
        for chunk, tokens, doc_length in zip(
            self._chunks, self._doc_tokens, self._doc_lengths, strict=False
        ):
            frequencies = Counter(tokens)
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                idf = self._idf.get(term, 0.0)
                denominator = frequency + self._k1 * (
                    1 - self._b + self._b * doc_length / max(self._avg_doc_length, 1)
                )
                score += idf * (frequency * (self._k1 + 1)) / denominator
            raw_scores[chunk.chunk_id] = score
        max_score = max(raw_scores.values(), default=0.0)
        if max_score <= 0:
            return {chunk_id: 0.0 for chunk_id in raw_scores}
        return {chunk_id: score / max_score for chunk_id, score in raw_scores.items()}


def _rank_map(chunks: list[RetrievedChunk], score_name: str) -> dict[str, int]:
    sorted_chunks = sorted(chunks, key=lambda chunk: getattr(chunk, score_name), reverse=True)
    return {chunk.chunk_id: index for index, chunk in enumerate(sorted_chunks, start=1)}


def _fused_score(
    chunk: RetrievedChunk,
    dense_rank: int | None,
    sparse_rank: int | None,
    dense_weight: float,
    sparse_weight: float,
    fusion_strategy: FusionStrategy,
    retrieval_mode: RetrievalMode,
    rrf_k: int,
) -> float:
    if retrieval_mode == "dense":
        return chunk.dense_score
    if retrieval_mode == "sparse":
        return chunk.sparse_score
    if fusion_strategy == "rrf":
        dense = 0.0 if dense_rank is None else dense_weight / (rrf_k + dense_rank)
        sparse = 0.0 if sparse_rank is None else sparse_weight / (rrf_k + sparse_rank)
        return dense + sparse
    return chunk.dense_score * dense_weight + chunk.sparse_score * sparse_weight


class InMemoryHybridRetriever(PolicyRetriever):
    def __init__(
        self,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        retrieval_mode: RetrievalMode = "hybrid",
        fusion_strategy: FusionStrategy = "weighted",
        candidate_multiplier: int = 4,
        rrf_k: int = 60,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight
        self._retrieval_mode = retrieval_mode
        self._fusion_strategy = fusion_strategy
        self._candidate_multiplier = max(candidate_multiplier, 1)
        self._rrf_k = rrf_k
        if embedding_model is None:
            raise ValueError("embedding_model is required")
        self._embeddings = embedding_model
        self._bm25_k1 = bm25_k1
        self._bm25_b = bm25_b

    async def retrieve(self, query: str, actor: ActorContext, top_k: int) -> list[RetrievedChunk]:
        start = time.perf_counter()
        labels = {"backend": "memory", "mode": self._retrieval_mode}
        try:
            candidate_limit = max(top_k * self._candidate_multiplier, top_k)
            query_vector = self._embeddings.embed_query(query)
            accessible_chunks = [chunk for chunk in POLICY_CHUNKS if _actor_can_access_chunk(actor, chunk)]
            sparse_scores = BM25SparseScorer(accessible_chunks, self._bm25_k1, self._bm25_b).scores(query)
            document_vectors = self._embeddings.embed_documents([chunk.text for chunk in accessible_chunks])
            candidates: list[RetrievedChunk] = []
            for source, document_vector in zip(accessible_chunks, document_vectors, strict=False):
                chunk = source.model_copy(deep=True)
                chunk.dense_score = cosine_score(query_vector, document_vector)
                chunk.sparse_score = sparse_scores.get(chunk.chunk_id, 0.0)
                candidates.append(chunk)

            dense_ranks = _rank_map(candidates, "dense_score")
            sparse_ranks = _rank_map(candidates, "sparse_score")
            for chunk in candidates:
                chunk.score = _fused_score(
                    chunk,
                    dense_ranks.get(chunk.chunk_id),
                    sparse_ranks.get(chunk.chunk_id),
                    self._dense_weight,
                    self._sparse_weight,
                    self._fusion_strategy,
                    self._retrieval_mode,
                    self._rrf_k,
                )
            results = sorted(candidates, key=lambda item: item.score, reverse=True)[:candidate_limit]
            observe_histogram(RAG_RETRIEVED_CHUNKS, labels, len(results))
            return results
        except Exception:
            increment(RAG_ERRORS, {"backend": "memory", "stage": "retrieve"})
            raise
        finally:
            observe_histogram(RAG_RETRIEVAL_SECONDS, labels, time.perf_counter() - start)


class MilvusHybridRetriever(PolicyRetriever):
    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "employee_policy_chunks",
        vector_dim: int = 64,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        retrieval_mode: RetrievalMode = "hybrid",
        fusion_strategy: FusionStrategy = "weighted",
        candidate_multiplier: int = 4,
        rrf_k: int = 60,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        fallback: PolicyRetriever | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._collection_name = collection_name
        self._vector_dim = vector_dim
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight
        self._retrieval_mode: RetrievalMode = retrieval_mode
        self._fusion_strategy: FusionStrategy = fusion_strategy
        self._candidate_multiplier = max(candidate_multiplier, 1)
        self._rrf_k = rrf_k
        if embedding_model is None:
            raise ValueError("embedding_model is required")
        self._embeddings = embedding_model
        self._bm25_k1 = bm25_k1
        self._bm25_b = bm25_b
        self._fallback = fallback

    async def retrieve(self, query: str, actor: ActorContext, top_k: int) -> list[RetrievedChunk]:
        if self._retrieval_mode == "sparse":
            if self._fallback is None:
                raise RuntimeError(
                    "Sparse-only Milvus search requires the native Milvus hybrid retriever"
                )
            return await self._fallback.retrieve(query, actor, top_k)
        start = time.perf_counter()
        labels = {"backend": "milvus_compat", "mode": self._retrieval_mode}
        candidate_limit = max(top_k * self._candidate_multiplier, top_k)
        try:
            collection = self._collection
            query_vector = self._embeddings.embed_query(query)
            hits = collection.search(
                data=[query_vector],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=candidate_limit,
                output_fields=["document_id", "title", "chunk_id", "text", "metadata_json"],
                expr=_milvus_access_expr(actor),
            )[0]
        except Exception:
            increment(RAG_ERRORS, {"backend": "milvus_compat", "stage": "search"})
            if self._fallback is None:
                raise
            return await self._fallback.retrieve(query, actor, top_k)

        sparse_scores = BM25SparseScorer(POLICY_CHUNKS, self._bm25_k1, self._bm25_b).scores(query)
        by_id: dict[str, RetrievedChunk] = {}
        for hit in hits:
            entity = hit.entity
            chunk_id = entity.get("chunk_id")
            dense = max(float(hit.score), 0.0)
            by_id[chunk_id] = RetrievedChunk(
                document_id=entity.get("document_id"),
                title=entity.get("title"),
                chunk_id=chunk_id,
                text=entity.get("text"),
                metadata=entity.get("metadata_json") or {},
                dense_score=dense,
                sparse_score=sparse_scores.get(chunk_id, 0.0),
            )

        sparse_candidates = sorted(
            (chunk for chunk in POLICY_CHUNKS if _actor_can_access_chunk(actor, chunk)),
            key=lambda chunk: sparse_scores.get(chunk.chunk_id, 0.0),
            reverse=True,
        )[:candidate_limit]
        for source in sparse_candidates:
            if source.chunk_id not in by_id:
                chunk = source.model_copy(deep=True)
                chunk.dense_score = 0.0
                chunk.sparse_score = sparse_scores.get(chunk.chunk_id, 0.0)
                by_id[chunk.chunk_id] = chunk

        candidates = list(by_id.values())
        dense_ranks = _rank_map(candidates, "dense_score")
        sparse_ranks = _rank_map(candidates, "sparse_score")
        for chunk in candidates:
            chunk.score = _fused_score(
                chunk,
                dense_ranks.get(chunk.chunk_id),
                sparse_ranks.get(chunk.chunk_id),
                self._dense_weight,
                self._sparse_weight,
                self._fusion_strategy,
                self._retrieval_mode,
                self._rrf_k,
            )
        results = sorted(candidates, key=lambda item: item.score, reverse=True)[:candidate_limit]
        observe_histogram(RAG_RETRIEVED_CHUNKS, labels, len(results))
        observe_histogram(RAG_RETRIEVAL_SECONDS, labels, time.perf_counter() - start)
        return results

    @cached_property
    def _collection(self):
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

        connections.connect(alias="default", host=self._host, port=str(self._port))
        if utility.has_collection(self._collection_name):
            existing = Collection(self._collection_name)
            field_names = {field.name for field in existing.schema.fields}
            dense_field = next((field for field in existing.schema.fields if field.name == "embedding"), None)
            dense_dim = int((getattr(dense_field, "params", {}) or {}).get("dim", 0)) if dense_field else 0
            expected_schema = {
                "embedding",
                "country_global",
                "country_code",
                "department_all",
                "department_code",
            }.issubset(field_names)
            if existing.num_entities != len(POLICY_CHUNKS) or dense_dim != self._vector_dim or not expected_schema:
                utility.drop_collection(self._collection_name)

        if not utility.has_collection(self._collection_name):
            fields = [
                FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
                FieldSchema(name="metadata_json", dtype=DataType.JSON),
                FieldSchema(name="country_global", dtype=DataType.BOOL),
                FieldSchema(name="country_code", dtype=DataType.VARCHAR, max_length=32),
                FieldSchema(name="department_all", dtype=DataType.BOOL),
                FieldSchema(name="department_code", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self._vector_dim),
            ]
            schema = CollectionSchema(fields=fields, description="Employee policy chunks")
            collection = Collection(self._collection_name, schema=schema)
            collection.create_index(
                field_name="embedding",
                index_params={"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 64}},
            )
            self._seed(collection)
        else:
            collection = Collection(self._collection_name)
            if collection.num_entities == 0:
                self._seed(collection)
        collection.load()
        return collection

    def rebuild_seed_collection(self) -> int:
        from pymilvus import Collection, connections, utility

        connections.connect(alias="default", host=self._host, port=str(self._port))
        if utility.has_collection(self._collection_name):
            utility.drop_collection(self._collection_name)
        self.__dict__.pop("_collection", None)
        collection: Collection = self._collection
        return int(collection.num_entities)

    def _seed(self, collection) -> None:
        rows = [
            {
                "document_id": chunk.document_id,
                "title": chunk.title,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata_json": chunk.metadata,
                **_access_fields(chunk),
                "embedding": vector,
            }
            for chunk, vector in zip(
                POLICY_CHUNKS,
                self._embeddings.embed_documents([chunk.text for chunk in POLICY_CHUNKS]),
                strict=False,
            )
        ]
        collection.insert(rows)
        collection.flush()


class NativeMilvusHybridRetriever(PolicyRetriever):
    """Milvus-native dense + sparse BM25 hybrid retriever.

    This stores both vector routes in Milvus:
    - dense: FLOAT_VECTOR generated by the app embedding adapter
    - sparse: SPARSE_FLOAT_VECTOR generated inside Milvus via BM25 Function

    Search is executed with Milvus hybrid_search plus WeightedRanker/RRFRanker.
    The older MilvusHybridRetriever remains as a compatibility fallback for
    Milvus deployments where BM25 full-text search is unavailable.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 19530,
        collection_name: str = "employee_policy_chunks",
        vector_dim: int = 64,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        retrieval_mode: RetrievalMode = "hybrid",
        fusion_strategy: FusionStrategy = "weighted",
        candidate_multiplier: int = 4,
        rrf_k: int = 60,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        fallback: PolicyRetriever | None = None,
        embedding_model: EmbeddingModel | None = None,
    ) -> None:
        self._uri = f"http://{host}:{port}"
        self._collection_name = collection_name
        self._vector_dim = vector_dim
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight
        self._retrieval_mode: RetrievalMode = retrieval_mode
        self._fusion_strategy: FusionStrategy = fusion_strategy
        self._candidate_multiplier = max(candidate_multiplier, 1)
        self._rrf_k = rrf_k
        if embedding_model is None:
            raise ValueError("embedding_model is required")
        self._embeddings = embedding_model
        self._fallback = fallback

    async def retrieve(self, query: str, actor: ActorContext, top_k: int) -> list[RetrievedChunk]:
        start = time.perf_counter()
        labels = {"backend": "milvus_native_hybrid", "mode": self._retrieval_mode}
        candidate_limit = max(top_k * self._candidate_multiplier, top_k)
        try:
            collection = self._collection
            requests = []
            if self._retrieval_mode in {"dense", "hybrid"}:
                requests.append(self._dense_request(query, candidate_limit, actor))
            if self._retrieval_mode in {"sparse", "hybrid"}:
                requests.append(self._sparse_request(query, candidate_limit, actor))
            ranker = self._ranker()
            hits = collection.hybrid_search(
                reqs=requests,
                rerank=ranker,
                limit=candidate_limit,
                output_fields=["document_id", "title", "chunk_id", "text", "metadata_json"],
            )[0]
        except Exception:
            increment(RAG_ERRORS, {"backend": "milvus_native_hybrid", "stage": "hybrid_search"})
            if self._fallback is None:
                raise
            return await self._fallback.retrieve(query, actor, top_k)

        chunks: list[RetrievedChunk] = []
        for hit in hits:
            entity = hit.entity
            chunks.append(
                RetrievedChunk(
                    document_id=entity.get("document_id"),
                    title=entity.get("title"),
                    chunk_id=entity.get("chunk_id"),
                    text=entity.get("text"),
                    metadata=(entity.get("metadata_json") or {})
                    | {
                        "retrieval_backend": "milvus_native_hybrid",
                        "retrieval_mode": self._retrieval_mode,
                        "fusion_strategy": self._fusion_strategy,
                    },
                    score=max(float(hit.score), 0.0),
                )
            )
        observe_histogram(RAG_RETRIEVED_CHUNKS, labels, len(chunks))
        observe_histogram(RAG_RETRIEVAL_SECONDS, labels, time.perf_counter() - start)
        return chunks

    @cached_property
    def _collection(self):
        from pymilvus import Collection, connections, utility

        connections.connect(alias="default", uri=self._uri)
        if utility.has_collection(self._collection_name):
            existing = Collection(self._collection_name)
            field_names = {field.name for field in existing.schema.fields}
            dense_field = next((field for field in existing.schema.fields if field.name == "dense"), None)
            dense_dim = int((getattr(dense_field, "params", {}) or {}).get("dim", 0)) if dense_field else 0
            expected_schema = {"dense", "sparse", "text", "country_global", "country_code", "department_all", "department_code"}.issubset(field_names)
            expected_dim = dense_dim == self._vector_dim
            expected_count = int(existing.num_entities) == len(POLICY_CHUNKS)
            if expected_schema and expected_dim and expected_count:
                existing.load()
                return existing
            utility.drop_collection(self._collection_name)

        collection = self._create_collection()
        self._seed(collection)
        collection.load()
        return collection

    def _create_collection(self):
        from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, Function, FunctionType

        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096, enable_analyzer=True),
            FieldSchema(name="metadata_json", dtype=DataType.JSON),
            FieldSchema(name="country_global", dtype=DataType.BOOL),
            FieldSchema(name="country_code", dtype=DataType.VARCHAR, max_length=32),
            FieldSchema(name="department_all", dtype=DataType.BOOL),
            FieldSchema(name="department_code", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="dense", dtype=DataType.FLOAT_VECTOR, dim=self._vector_dim),
            FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),
        ]
        schema = CollectionSchema(fields=fields, description="Employee policy chunks")
        schema.add_function(
            Function(
                name="text_bm25",
                input_field_names=["text"],
                output_field_names=["sparse"],
                function_type=FunctionType.BM25,
            )
        )
        collection = Collection(self._collection_name, schema=schema)
        collection.create_index(
            field_name="dense",
            index_params={"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 64}},
        )
        collection.create_index(
            field_name="sparse",
            index_params={"metric_type": "BM25", "index_type": "AUTOINDEX", "params": {}},
        )
        return collection

    def _dense_request(self, query: str, limit: int, actor: ActorContext):
        from pymilvus import AnnSearchRequest

        return AnnSearchRequest(
            data=[self._embeddings.embed_query(query)],
            anns_field="dense",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=limit,
            expr=_milvus_access_expr(actor),
        )

    def _sparse_request(self, query: str, limit: int, actor: ActorContext):
        from pymilvus import AnnSearchRequest

        return AnnSearchRequest(
            data=[query],
            anns_field="sparse",
            param={"metric_type": "BM25", "params": {}},
            limit=limit,
            expr=_milvus_access_expr(actor),
        )

    def _ranker(self):
        from pymilvus import RRFRanker, WeightedRanker

        if self._fusion_strategy == "rrf":
            return RRFRanker(self._rrf_k)
        if self._retrieval_mode == "dense":
            return WeightedRanker(1.0)
        if self._retrieval_mode == "sparse":
            return WeightedRanker(1.0)
        return WeightedRanker(self._dense_weight, self._sparse_weight)

    def rebuild_seed_collection(self) -> int:
        from pymilvus import connections, utility

        connections.connect(alias="default", uri=self._uri)
        if utility.has_collection(self._collection_name):
            utility.drop_collection(self._collection_name)
        self.__dict__.pop("_collection", None)
        collection = self._collection
        return int(collection.num_entities)

    def _seed(self, collection) -> None:
        rows = [
            {
                "document_id": chunk.document_id,
                "title": chunk.title,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata_json": chunk.metadata,
                **_access_fields(chunk),
                "dense": vector,
            }
            for chunk, vector in zip(
                POLICY_CHUNKS,
                self._embeddings.embed_documents([chunk.text for chunk in POLICY_CHUNKS]),
                strict=False,
            )
        ]
        collection.insert(rows)
        collection.flush()


class ScoreReranker(Reranker):
    async def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        start = time.perf_counter()
        try:
            return sorted(chunks, key=lambda item: item.score, reverse=True)
        finally:
            observe_histogram(RAG_RERANK_SECONDS, {"provider": "score"}, time.perf_counter() - start)


class HeuristicCrossEncoderReranker(Reranker):
    def __init__(self, base_weight: float = 0.7, semantic_weight: float = 0.3) -> None:
        self._base_weight = base_weight
        self._semantic_weight = semantic_weight

    async def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        start = time.perf_counter()
        try:
            for chunk in chunks:
                semantic = lexical_score(query, f"{chunk.title} {chunk.text}")
                chunk.metadata = chunk.metadata | {"reranker": "heuristic_cross_encoder", "rerank_score": semantic}
                chunk.score = chunk.score * self._base_weight + semantic * self._semantic_weight
            return sorted(chunks, key=lambda item: item.score, reverse=True)
        finally:
            observe_histogram(RAG_RERANK_SECONDS, {"provider": "heuristic_cross_encoder"}, time.perf_counter() - start)
