from fastapi.testclient import TestClient

from app.api.dependencies import get_agent
from app.main import app
from tests.support_embeddings import SupportEmbeddingModel


def test_chat_api_policy_smoke(agent):
    app.dependency_overrides[get_agent] = lambda: agent
    client = TestClient(app)
    response = client.post(
        "/api/v1/chat",
        headers={"Authorization": "Bearer dev-token"},
        json={"message": "What is the leave policy?"},
    )

    assert response.status_code == 200
    assert response.json()["intent"] == "policy_qa"
    app.dependency_overrides.clear()



def test_ops_knowledge_ingest_and_eval_smoke(monkeypatch):
    monkeypatch.setattr(
        "app.application.operations._build_embedding_model",
        lambda settings: SupportEmbeddingModel(),
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer dev-token"}

    status = client.get("/api/v1/ops/knowledge", headers=headers)
    assert status.status_code == 200
    assert status.json()["corpus_chunks"] >= 60
    assert status.json()["eval_cases"] >= 60

    ingest = client.post("/api/v1/ops/ingest", headers=headers)
    assert ingest.status_code == 200
    assert ingest.json()["indexed_chunks"] >= 60

    eval_response = client.post("/api/v1/ops/eval", headers=headers)
    assert eval_response.status_code == 200
    summary = eval_response.json()["summary"]
    assert summary["cases"] >= 60
    assert "ragas_context_precision" in summary
    assert "ragas_context_recall" in summary


def test_metrics_exposes_rag_and_agent_metrics(agent):
    app.dependency_overrides[get_agent] = lambda: agent
    client = TestClient(app)
    response = client.post(
        "/api/v1/chat",
        headers={"Authorization": "Bearer dev-token"},
        json={"message": "What is the leave policy?"},
    )
    assert response.status_code == 200

    metrics = client.get("/metrics")

    assert metrics.status_code == 200
    assert "employee_support_agent_node_seconds" in metrics.text
    assert "employee_support_rag_retrieval_seconds" in metrics.text
    assert "employee_support_llm_seconds" in metrics.text
    app.dependency_overrides.clear()
