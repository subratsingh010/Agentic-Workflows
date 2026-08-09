# Architecture

The system is a single agent with explicit graph nodes and framework-independent business rules. FastAPI only handles HTTP concerns. Tool clients, retrieval, authorization, persistence, rate limiting, and LLM behavior are injected through ports.

```mermaid
flowchart LR
  UI["React chat UI"] --> API["FastAPI API"]
  API --> Graph["Single LangGraph-style agent"]
  Graph --> Auth["Keycloak JWT auth"]
  Graph --> Policy["RBAC/ABAC policy"]
  Graph --> RAG["Milvus hybrid RAG"]
  Graph --> Tools["Typed Jira/Leave mock tools"]
  Graph --> DB["PostgreSQL conversations/checkpoints/audit"]
  Worker["Celery ingestion"] --> RAG
  API --> OTel["OpenTelemetry/Phoenix/Prometheus"]
```

