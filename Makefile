.PHONY: test backend frontend up down lint eval eval-live eval-compare ingest-build

test:
	cd backend && pytest

backend:
	cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

frontend:
	cd frontend && npm run dev -- --host 0.0.0.0

up:
	docker compose up --build

down:
	docker compose down -v

lint:
	ruff check backend



eval:
	conda run -n agent python backend/scripts/run_policy_eval.py --top-k 5 --min-hit-at-k 0.70 --min-document-hit-at-k 0.80 --min-ndcg-at-k 0.50 --min-keyword-coverage 0.70

eval-live:
	conda run -n agent python backend/scripts/run_policy_eval.py --live-api http://localhost:8000 --top-k 5


ingest-build:
	conda run -n agent python backend/scripts/build_policy_seed.py


eval-compare: ingest-build
	conda run -n agent python backend/scripts/run_policy_eval.py --top-k 5 --compare-retrieval-modes
