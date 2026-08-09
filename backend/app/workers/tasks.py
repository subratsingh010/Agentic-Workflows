from app.application.operations import ingest_seed_corpus
from app.core.config import get_settings
from app.workers.celery_app import celery_app


@celery_app.task(
    bind=True,
    autoretry_for=(RuntimeError,),
    retry_backoff=True,
    retry_kwargs={"max_retries": 3},
)
def ingest_policy_document(self, document_id: str, path: str) -> dict[str, str | int]:
    result = ingest_seed_corpus(get_settings())
    return {"document_id": document_id, "path": path, **result}

