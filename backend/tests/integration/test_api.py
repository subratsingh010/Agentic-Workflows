from fastapi.testclient import TestClient

from app.main import app


def test_chat_api_policy_smoke():
    client = TestClient(app)
    response = client.post(
        "/api/v1/chat",
        headers={"Authorization": "Bearer dev-token"},
        json={"message": "What is the leave policy?"},
    )

    assert response.status_code == 200
    assert response.json()["intent"] == "policy_qa"



def test_ops_knowledge_ingest_and_eval_smoke():
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
    assert eval_response.json()["summary"]["cases"] >= 60
