import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { Activity, Database, Play, RefreshCw, Send, ShieldCheck } from "lucide-react";
import { fetchKnowledgeStatus, ingestSeedCorpus, rebuildAndIngest, runPolicyEval, sendMessage } from "./api/client";
import type { ChatResponse, EvalRunResponse, KnowledgeStatus, RagOpsConfig } from "./types/chat";
import "./styles.css";

type Message = {
  role: "user" | "assistant";
  content: string;
  response?: ChatResponse;
};

function pct(value: number | undefined): string {
  if (typeof value !== "number") return "--";
  return `${Math.round(value * 100)}%`;
}

function defaultOpsConfig(knowledge: KnowledgeStatus | null): RagOpsConfig {
  return {
    chunk_size: knowledge?.chunk_size ?? 120,
    chunk_overlap: knowledge?.chunk_overlap ?? 20,
    retrieval_mode: (knowledge?.retrieval_mode as RagOpsConfig["retrieval_mode"]) ?? "hybrid",
    fusion_strategy: (knowledge?.fusion_strategy as RagOpsConfig["fusion_strategy"]) ?? "weighted",
    reranker_provider: (knowledge?.reranker_provider as RagOpsConfig["reranker_provider"]) ?? "score",
    milvus_native_hybrid: knowledge?.milvus_native_hybrid ?? true,
    top_k: knowledge?.top_k ?? 5,
    dense_weight: knowledge?.dense_weight ?? 0.6,
    sparse_weight: knowledge?.sparse_weight ?? 0.4,
    candidate_multiplier: knowledge?.candidate_multiplier ?? 4,
    embedding_provider: knowledge?.embedding_provider ?? "sentence_transformers",
    embedding_model: knowledge?.embedding_model ?? "BAAI/bge-small-en-v1.5",
    embedding_device: knowledge?.embedding_device ?? "cpu",
    embedding_dim: knowledge?.embedding_dim ?? 384
  };
}

function App() {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      content: "Ask about policy, Jira time logs, or leave. Dates work best as YYYY-MM-DD."
    }
  ]);
  const [draft, setDraft] = useState("");
  const [threadId, setThreadId] = useState<string | undefined>();
  const [pendingConfirmation, setPendingConfirmation] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [opsBusy, setOpsBusy] = useState(false);
  const [opsMessage, setOpsMessage] = useState("Ready");
  const [knowledge, setKnowledge] = useState<KnowledgeStatus | null>(null);
  const [opsConfig, setOpsConfig] = useState<RagOpsConfig>(defaultOpsConfig(null));
  const [lastEval, setLastEval] = useState<EvalRunResponse | null>(null);

  async function refreshKnowledge() {
    const status = await fetchKnowledgeStatus();
    setKnowledge(status);
    setOpsConfig((current) => ({ ...defaultOpsConfig(status), ...current }));
  }

  useEffect(() => {
    refreshKnowledge().catch((error) => {
      setOpsMessage(error instanceof Error ? error.message : "Status unavailable");
    });
  }, []);

  function updateOpsConfig<K extends keyof RagOpsConfig>(key: K, value: RagOpsConfig[K]) {
    setOpsConfig((current) => ({ ...current, [key]: value }));
  }

  async function submit(confirm = false) {
    const message = confirm && pendingConfirmation ? pendingConfirmation : draft.trim();
    if (!message || busy) return;
    setBusy(true);
    setDraft("");
    if (!confirm) setMessages((items) => [...items, { role: "user", content: message }]);
    try {
      const response = await sendMessage({ message, threadId, confirm });
      setThreadId(response.thread_id);
      setPendingConfirmation(response.requires_confirmation ? message : null);
      setMessages((items) => [
        ...items,
        { role: "assistant", content: response.answer, response }
      ]);
    } catch (error) {
      setMessages((items) => [
        ...items,
        { role: "assistant", content: error instanceof Error ? error.message : "Request failed" }
      ]);
    } finally {
      setBusy(false);
    }
  }

  async function runOps(action: "ingest" | "rebuild" | "eval" | "refresh") {
    if (opsBusy) return;
    setOpsBusy(true);
    try {
      if (action === "ingest") {
        const result = await ingestSeedCorpus(opsConfig);
        setOpsMessage(`Ingested ${result.indexed_chunks} chunks into ${result.rag_backend}`);
      } else if (action === "rebuild") {
        const result = await rebuildAndIngest(opsConfig);
        setOpsMessage(`Rebuilt ${result.rebuild.chunks} chunks and ingested ${result.ingest.indexed_chunks}`);
      } else if (action === "eval") {
        const result = await runPolicyEval(opsConfig);
        const hitAtK = result.summary[`hit_at_${result.summary.top_k}`] as number | undefined;
        setLastEval(result);
        setOpsMessage(`Eval complete: hit@${result.summary.top_k} ${pct(hitAtK)}`);
      } else {
        setOpsMessage("Status refreshed");
      }
      await refreshKnowledge();
    } catch (error) {
      setOpsMessage(error instanceof Error ? error.message : "Operation failed");
    } finally {
      setOpsBusy(false);
    }
  }

  const summary = knowledge?.latest_eval;
  const hitAtK = summary ? (summary[`hit_at_${summary.top_k}`] as number | undefined) : undefined;
  const precisionAtK = summary ? (summary[`precision_at_${summary.top_k}`] as number | undefined) : undefined;
  const ndcgAtK = summary ? (summary[`ndcg_at_${summary.top_k}`] as number | undefined) : undefined;

  return (
    <main className="min-h-screen bg-zinc-50 text-zinc-950">
      <header className="border-b border-zinc-200 bg-white">
        <div className="mx-auto flex max-w-6xl items-center gap-3 px-4 py-4">
          <ShieldCheck className="h-6 w-6 text-emerald-600" />
          <div>
            <h1 className="text-lg font-semibold">Employee Support</h1>
            <p className="text-sm text-zinc-500">Single-agent support console</p>
          </div>
        </div>
      </header>
      <section className="mx-auto grid max-w-6xl gap-4 px-4 py-6 lg:grid-cols-[1fr_360px]">
        <div className="flex min-w-0 flex-col gap-4">
          <div className="h-[62vh] overflow-y-auto rounded border border-zinc-200 bg-white p-4">
            {messages.map((message, index) => (
              <article key={index} className={`mb-4 ${message.role === "user" ? "text-right" : ""}`}>
                <div
                  className={`inline-block max-w-[82%] rounded px-3 py-2 text-sm leading-6 ${
                    message.role === "user" ? "bg-zinc-900 text-white" : "bg-zinc-100 text-zinc-900"
                  }`}
                >
                  {message.content}
                </div>
                {message.response?.citations.length ? (
                  <div className="mt-2 space-y-1 text-left text-xs text-zinc-500">
                    {message.response.citations.map((citation) => (
                      <div key={citation.chunk_id}>
                        {citation.title} · score {citation.score.toFixed(2)}
                      </div>
                    ))}
                  </div>
                ) : null}
              </article>
            ))}
          </div>
          {pendingConfirmation ? (
            <button
              className="w-fit rounded bg-emerald-600 px-3 py-2 text-sm font-medium text-white"
              onClick={() => submit(true)}
            >
              Confirm leave submission
            </button>
          ) : null}
          <form
            className="flex gap-2"
            onSubmit={(event) => {
              event.preventDefault();
              submit();
            }}
          >
            <input
              className="min-h-11 flex-1 rounded border border-zinc-300 px-3 outline-none focus:border-zinc-900"
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              placeholder="Ask a policy question or request an action"
            />
            <button
              className="grid h-11 w-11 place-items-center rounded bg-zinc-900 text-white disabled:opacity-50"
              disabled={busy}
              aria-label="Send"
            >
              <Send className="h-5 w-5" />
            </button>
          </form>
        </div>

        <aside className="rounded border border-zinc-200 bg-white p-4">
          <div className="mb-4 flex items-center gap-2">
            <Activity className="h-5 w-5 text-emerald-600" />
            <h2 className="text-sm font-semibold">Knowledge & Eval</h2>
          </div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <div className="rounded border border-zinc-200 p-3">
              <div className="text-xs text-zinc-500">Corpus</div>
              <div className="text-lg font-semibold">{knowledge?.corpus_chunks ?? "--"}</div>
            </div>
            <div className="rounded border border-zinc-200 p-3">
              <div className="text-xs text-zinc-500">Eval Cases</div>
              <div className="text-lg font-semibold">{knowledge?.eval_cases ?? "--"}</div>
            </div>
            <div className="rounded border border-zinc-200 p-3">
              <div className="text-xs text-zinc-500">Hit@K</div>
              <div className="text-lg font-semibold">{pct(hitAtK)}</div>
            </div>
            <div className="rounded border border-zinc-200 p-3">
              <div className="text-xs text-zinc-500">Doc Hit</div>
              <div className="text-lg font-semibold">{pct(summary?.document_hit_at_k)}</div>
            </div>
            <div className="rounded border border-zinc-200 p-3">
              <div className="text-xs text-zinc-500">MRR</div>
              <div className="text-lg font-semibold">{summary?.mrr.toFixed(2) ?? "--"}</div>
            </div>
            <div className="rounded border border-zinc-200 p-3">
              <div className="text-xs text-zinc-500">Precision@K</div>
              <div className="text-lg font-semibold">{pct(precisionAtK)}</div>
            </div>
            <div className="rounded border border-zinc-200 p-3">
              <div className="text-xs text-zinc-500">nDCG@K</div>
              <div className="text-lg font-semibold">{pct(ndcgAtK)}</div>
            </div>
            <div className="rounded border border-zinc-200 p-3">
              <div className="text-xs text-zinc-500">Keywords</div>
              <div className="text-lg font-semibold">{pct(summary?.keyword_coverage)}</div>
            </div>
          </div>

          <div className="mt-4 space-y-3 rounded border border-zinc-200 p-3 text-sm">
            <div className="grid grid-cols-2 gap-2">
              <label className="text-xs text-zinc-500">
                Chunk Size
                <input className="mt-1 w-full rounded border border-zinc-300 px-2 py-1 text-sm text-zinc-900" type="number" min="20" value={opsConfig.chunk_size} onChange={(event) => updateOpsConfig("chunk_size", Number(event.target.value))} />
              </label>
              <label className="text-xs text-zinc-500">
                Overlap
                <input className="mt-1 w-full rounded border border-zinc-300 px-2 py-1 text-sm text-zinc-900" type="number" min="0" value={opsConfig.chunk_overlap} onChange={(event) => updateOpsConfig("chunk_overlap", Number(event.target.value))} />
              </label>
              <label className="text-xs text-zinc-500">
                Top K
                <input className="mt-1 w-full rounded border border-zinc-300 px-2 py-1 text-sm text-zinc-900" type="number" min="1" max="20" value={opsConfig.top_k} onChange={(event) => updateOpsConfig("top_k", Number(event.target.value))} />
              </label>
              <label className="text-xs text-zinc-500">
                Candidates
                <input className="mt-1 w-full rounded border border-zinc-300 px-2 py-1 text-sm text-zinc-900" type="number" min="1" max="20" value={opsConfig.candidate_multiplier} onChange={(event) => updateOpsConfig("candidate_multiplier", Number(event.target.value))} />
              </label>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <label className="text-xs text-zinc-500">
                Retrieval
                <select className="mt-1 w-full rounded border border-zinc-300 px-2 py-1 text-sm text-zinc-900" value={opsConfig.retrieval_mode} onChange={(event) => updateOpsConfig("retrieval_mode", event.target.value as RagOpsConfig["retrieval_mode"])}>
                  <option value="hybrid">hybrid</option>
                  <option value="dense">dense</option>
                  <option value="sparse">sparse</option>
                </select>
              </label>
              <label className="text-xs text-zinc-500">
                Fusion
                <select className="mt-1 w-full rounded border border-zinc-300 px-2 py-1 text-sm text-zinc-900" value={opsConfig.fusion_strategy} onChange={(event) => updateOpsConfig("fusion_strategy", event.target.value as RagOpsConfig["fusion_strategy"])}>
                  <option value="weighted">weighted</option>
                  <option value="rrf">rrf</option>
                </select>
              </label>
              <label className="text-xs text-zinc-500">
                Dense Weight
                <input className="mt-1 w-full rounded border border-zinc-300 px-2 py-1 text-sm text-zinc-900" type="number" min="0" max="1" step="0.1" value={opsConfig.dense_weight} onChange={(event) => updateOpsConfig("dense_weight", Number(event.target.value))} />
              </label>
              <label className="text-xs text-zinc-500">
                Sparse Weight
                <input className="mt-1 w-full rounded border border-zinc-300 px-2 py-1 text-sm text-zinc-900" type="number" min="0" max="1" step="0.1" value={opsConfig.sparse_weight} onChange={(event) => updateOpsConfig("sparse_weight", Number(event.target.value))} />
              </label>
            </div>
            <label className="block text-xs text-zinc-500">
              Reranker
              <select className="mt-1 w-full rounded border border-zinc-300 px-2 py-1 text-sm text-zinc-900" value={opsConfig.reranker_provider} onChange={(event) => updateOpsConfig("reranker_provider", event.target.value as RagOpsConfig["reranker_provider"])}>
                <option value="score">score</option>
                <option value="heuristic_cross_encoder">heuristic cross-encoder</option>
              </select>
            </label>
            <div className="grid grid-cols-2 gap-2">
              <label className="text-xs text-zinc-500">
                Embedding
                <select className="mt-1 w-full rounded border border-zinc-300 px-2 py-1 text-sm text-zinc-900" value={opsConfig.embedding_provider} onChange={(event) => updateOpsConfig("embedding_provider", event.target.value as RagOpsConfig["embedding_provider"])}>
                  <option value="sentence_transformers">BGE</option>
                </select>
              </label>
              <label className="text-xs text-zinc-500">
                Emb Dim
                <input className="mt-1 w-full rounded border border-zinc-300 px-2 py-1 text-sm text-zinc-900" type="number" min="16" max="4096" value={opsConfig.embedding_dim} onChange={(event) => updateOpsConfig("embedding_dim", Number(event.target.value))} />
              </label>
            </div>
            <label className="flex items-center gap-2 text-xs text-zinc-600">
              <input type="checkbox" checked={opsConfig.milvus_native_hybrid} onChange={(event) => updateOpsConfig("milvus_native_hybrid", event.target.checked)} />
              Native Milvus hybrid
            </label>
          </div>

          <div className="mt-4 flex flex-wrap gap-2">
            <button
              className="inline-flex items-center gap-2 rounded border border-zinc-300 px-3 py-2 text-sm disabled:opacity-50"
              disabled={opsBusy}
              onClick={() => runOps("rebuild")}
            >
              <Database className="h-4 w-4" />
              Rebuild + Ingest
            </button>
            <button
              className="inline-flex items-center gap-2 rounded border border-zinc-300 px-3 py-2 text-sm disabled:opacity-50"
              disabled={opsBusy}
              onClick={() => runOps("ingest")}
            >
              <Database className="h-4 w-4" />
              Ingest
            </button>
            <button
              className="inline-flex items-center gap-2 rounded bg-zinc-900 px-3 py-2 text-sm text-white disabled:opacity-50"
              disabled={opsBusy}
              onClick={() => runOps("eval")}
            >
              <Play className="h-4 w-4" />
              Run Eval
            </button>
            <button
              className="grid h-9 w-9 place-items-center rounded border border-zinc-300 disabled:opacity-50"
              disabled={opsBusy}
              onClick={() => runOps("refresh")}
              aria-label="Refresh status"
            >
              <RefreshCw className="h-4 w-4" />
            </button>
          </div>
          <p className="mt-3 text-xs leading-5 text-zinc-500">{opsMessage}</p>
          {lastEval ? (
            <div className="mt-3 space-y-3 border-t border-zinc-200 pt-3 text-xs text-zinc-600">
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <div className="text-zinc-400">Embedding</div>
                  <div className="font-medium text-zinc-800">{lastEval.pipeline.embedding.model}</div>
                  <div>{lastEval.pipeline.embedding.dim} dim · {lastEval.pipeline.embedding.device}</div>
                </div>
                <div>
                  <div className="text-zinc-400">Milvus</div>
                  <div className="font-medium text-zinc-800">{lastEval.pipeline.retrieval.native_milvus_hybrid ? "native hybrid" : lastEval.pipeline.retrieval.backend}</div>
                  <div>{lastEval.pipeline.retrieval.mode} · {lastEval.pipeline.retrieval.fusion}</div>
                </div>
                <div>
                  <div className="text-zinc-400">Reranker</div>
                  <div className="font-medium text-zinc-800">{lastEval.pipeline.reranker.provider}</div>
                </div>
                <div>
                  <div className="text-zinc-400">LLM</div>
                  <div className="font-medium text-zinc-800">{lastEval.pipeline.llm.provider}</div>
                  <div>{lastEval.pipeline.llm.model}</div>
                </div>
              </div>
              {lastEval.cases[0] ? (
                <div className="rounded border border-zinc-200 p-2">
                  <div className="font-medium text-zinc-800">Sample Eval Case</div>
                  <div className="mt-1 line-clamp-2">{lastEval.cases[0].question}</div>
                  <div className="mt-1 text-zinc-400">Top chunks: {(lastEval.cases[0].post_rerank_chunk_ids ?? lastEval.cases[0].retrieved_chunk_ids).slice(0, 3).join(", ")}</div>
                  <div className="mt-1 line-clamp-3">{lastEval.cases[0].generated_answer}</div>
                </div>
              ) : null}
            </div>
          ) : null}
          <p className="mt-2 text-xs leading-5 text-zinc-400">Collection: {knowledge?.collection ?? "--"}</p>
          <p className="mt-1 truncate text-xs leading-5 text-zinc-400">Source: {knowledge?.source_dir ?? "--"}</p>
        </aside>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
