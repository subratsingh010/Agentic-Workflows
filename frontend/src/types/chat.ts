export type Intent =
  | "policy_qa"
  | "jira_time_log_lookup"
  | "leave_application"
  | "small_talk"
  | "unsupported_tool"
  | "unknown";

export type Citation = {
  document_id: string;
  title: string;
  chunk_id: string;
  score: number;
  excerpt: string;
};

export type ChatResponse = {
  thread_id: string;
  message_id: string;
  intent: Intent;
  answer: string;
  citations: Citation[];
  requires_confirmation: boolean;
  missing_fields: string[];
  tool_result: Record<string, unknown> | null;
};

export type EvalSummary = {
  cases: number;
  top_k: number;
  hit_at_1: number;
  document_hit_at_k: number;
  mrr: number;
  keyword_coverage?: number;
  llm_judge_score?: number;
  answer_correctness?: number;
  answer_groundedness?: number;
  [key: string]: unknown;
};

export type KnowledgeStatus = {
  rag_backend: string;
  corpus_chunks: number;
  eval_cases: number;
  collection: string;
  latest_eval: EvalSummary | null;
  chunk_size: number;
  chunk_overlap: number;
  source_dir: string;
  retrieval_mode: string;
  fusion_strategy: string;
  reranker_provider: string;
  candidate_multiplier: number;
  dense_weight: number;
  sparse_weight: number;
  top_k: number;
  milvus_native_hybrid: boolean;
  embedding_provider: "sentence_transformers";
  embedding_model: string;
  embedding_device?: string;
  embedding_dim: number;
};

export type RagOpsConfig = {
  chunk_size: number;
  chunk_overlap: number;
  retrieval_mode: "dense" | "sparse" | "hybrid";
  fusion_strategy: "weighted" | "rrf";
  reranker_provider: "score" | "heuristic_cross_encoder";
  milvus_native_hybrid: boolean;
  top_k: number;
  dense_weight: number;
  sparse_weight: number;
  candidate_multiplier: number;
  embedding_provider: "sentence_transformers";
  embedding_model: string;
  embedding_device: string;
  embedding_dim: number;
};

export type OpsIngestResponse = {
  status: string;
  rag_backend: string;
  indexed_chunks: number;
  source: string;
};

export type OpsRebuildResponse = {
  rebuild: {
    status: string;
    chunks: number;
    chunk_size: number;
    chunk_overlap: number;
    source: string;
  };
  ingest: OpsIngestResponse;
};

export type EvalPipeline = {
  embedding: {
    provider: string;
    model: string;
    device: string;
    dim: number;
  };
  ingestion: {
    source_dir: string;
    chunk_size: number;
    chunk_overlap: number;
    chunks: number;
  };
  retrieval: {
    backend: string;
    native_milvus_hybrid: boolean;
    collection: string;
    mode: string;
    fusion: string;
    dense_weight: number;
    sparse_weight: number;
    candidate_multiplier: number;
  };
  reranker: {
    enabled: boolean;
    provider: string;
  };
  llm: {
    provider: string;
    model: string;
  };
};

export type EvalCaseResult = {
  id: string;
  question: string;
  expected_chunk_id: string;
  retrieved_chunk_ids: string[];
  pre_rerank_chunk_ids?: string[];
  post_rerank_chunk_ids?: string[];
  generated_answer?: string;
  ground_truth?: string;
  retrieval_backend?: string;
  reranker?: string;
  rank?: number | null;
  top_score?: number;
  [key: string]: unknown;
};

export type EvalRunResponse = {
  summary: EvalSummary;
  pipeline: EvalPipeline;
  cases: EvalCaseResult[];
  output: string;
};
