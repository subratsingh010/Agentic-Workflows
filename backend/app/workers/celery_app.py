from celery import Celery
from kombu import Exchange, Queue

from app.core.config import get_settings

settings = get_settings()
celery_app = Celery(
    "employee_support",
    broker=settings.rabbitmq_url,
    backend=settings.redis_url,
)
dead_letter_exchange = Exchange("ingestion.dlx", type="direct")
ingestion_exchange = Exchange("ingestion", type="direct")
celery_app.conf.task_default_queue = "ingestion"
celery_app.conf.task_default_exchange = "ingestion"
celery_app.conf.task_default_routing_key = "ingestion"
celery_app.conf.task_acks_late = True
celery_app.conf.task_reject_on_worker_lost = True
celery_app.conf.task_time_limit = 600
celery_app.conf.task_soft_time_limit = 540
celery_app.conf.worker_prefetch_multiplier = 1
celery_app.conf.task_queues = (
    Queue(
        "ingestion",
        ingestion_exchange,
        routing_key="ingestion",
        queue_arguments={
            "x-dead-letter-exchange": "ingestion.dlx",
            "x-dead-letter-routing-key": "ingestion.dlq",
        },
    ),
    Queue("ingestion.dlq", dead_letter_exchange, routing_key="ingestion.dlq"),
)
celery_app.conf.task_routes = {
    "app.workers.tasks.ingest_policy_document": {"queue": "ingestion", "routing_key": "ingestion"},
}

