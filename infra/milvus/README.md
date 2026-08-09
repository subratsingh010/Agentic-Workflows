# Milvus

The `MilvusHybridRetriever` port is shaped for dense+sparse hybrid search with configurable `top_k`, `dense_weight`, `sparse_weight`, optional reranking, and citation return. The scaffold uses synthetic in-process data so tests run without services; replace the adapter body with `pymilvus` collection search/upsert in production.

