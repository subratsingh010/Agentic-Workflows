# Employee Support AI Agent Repository Summary

Created a complete runnable monorepo scaffold at:

`/Users/subrat/Documents/Codex/2026-08-08/referenced-chatgpt-conversation-this-is-an`

Included:

- Python/FastAPI backend with a single LangGraph-backed agent
- Explicit workflow nodes for validation, JWT auth, ActorContext, authorization, intent, missing fields, RAG, reranking, generation, typed tools, confirmation, idempotency, audit, and response
- Clean architecture ports/adapters for auth, authorization, persistence, RAG, LLM, Jira, Leave, idempotency, rate limiting, and audit
- PostgreSQL schema migration and in-memory local/test adapters
- Mock Jira and Leave REST-style typed clients
- Milvus hybrid RAG adapter shape with synthetic fallback data
- Celery ingestion worker scaffold with retry behavior
- Docker Compose for backend, worker, frontend, PostgreSQL, Redis, RabbitMQ, Milvus, Keycloak, Prometheus, Grafana, Loki, Tempo, and Phoenix
- React + TypeScript + Vite + Tailwind chat UI
- Unit, integration, security, E2E, and Ragas evaluation scaffold tests
- README, architecture diagram, `.env.example`, observability and security notes

Validation performed:

- `python3 -m compileall -q backend/app backend/tests`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -p pytest_asyncio.plugin -q backend/tests`

Result:

- 10 tests passed
- 1 LangGraph deprecation warning from an upstream package

