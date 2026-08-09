import type {
  ChatResponse,
  EvalRunResponse,
  KnowledgeStatus,
  OpsIngestResponse,
  OpsRebuildResponse,
  RagOpsConfig
} from "../types/chat";

export async function sendMessage(params: {
  message: string;
  threadId?: string;
  confirm?: boolean;
}): Promise<ChatResponse> {
  const response = await fetch("/api/v1/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: "Bearer dev-token"
    },
    body: JSON.stringify({
      message: params.message,
      thread_id: params.threadId,
      confirm: params.confirm ?? false,
      idempotency_key: crypto.randomUUID()
    })
  });
  if (!response.ok) {
    const raw = await response.text();
    try {
      const payload = JSON.parse(raw) as { detail?: unknown };
      throw new Error(typeof payload.detail === "string" ? payload.detail : raw);
    } catch (error) {
      if (error instanceof Error && error.message !== raw) throw error;
      throw new Error(raw || `Request failed with ${response.status}`);
    }
  }
  return response.json();
}

async function opsFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      Authorization: "Bearer dev-token",
      ...(init?.headers ?? {})
    }
  });
  if (!response.ok) {
    const raw = await response.text();
    try {
      const payload = JSON.parse(raw) as { detail?: unknown };
      throw new Error(typeof payload.detail === "string" ? payload.detail : raw);
    } catch (error) {
      if (error instanceof Error && error.message !== raw) throw error;
      throw new Error(raw || `Request failed with ${response.status}`);
    }
  }
  return response.json();
}

function configBody(config: RagOpsConfig): RequestInit {
  return { method: "POST", body: JSON.stringify(config) };
}

export async function fetchKnowledgeStatus(): Promise<KnowledgeStatus> {
  return opsFetch<KnowledgeStatus>("/api/v1/ops/knowledge");
}

export async function ingestSeedCorpus(config?: RagOpsConfig): Promise<OpsIngestResponse> {
  return opsFetch<OpsIngestResponse>("/api/v1/ops/ingest", config ? configBody(config) : { method: "POST" });
}

export async function rebuildAndIngest(config: RagOpsConfig): Promise<OpsRebuildResponse> {
  return opsFetch<OpsRebuildResponse>("/api/v1/ops/rebuild", configBody(config));
}

export async function runPolicyEval(config?: RagOpsConfig): Promise<EvalRunResponse> {
  return opsFetch<EvalRunResponse>("/api/v1/ops/eval", config ? configBody(config) : { method: "POST" });
}
