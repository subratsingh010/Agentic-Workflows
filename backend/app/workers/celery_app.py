from celery import Celery

from app.core.config import get_settings

settings = get_settings()
celery_app = Celery(
    "employee_support",
    broker=settings.rabbitmq_url,
    backend=settings.redis_url,
)
celery_app.conf.task_default_queue = "ingestion"
celery_app.conf.task_acks_late = True
celery_app.conf.task_reject_on_worker_lost = True
celery_app.conf.task_routes = {
    "app.workers.tasks.ingest_policy_document": {"queue": "ingestion"},
}

