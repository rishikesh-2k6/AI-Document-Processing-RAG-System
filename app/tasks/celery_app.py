"""Celery application factory and configuration."""

from celery import Celery

from app.core.config import settings

celery_app = Celery(
    "rag_tasks",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["app.tasks.ingest_task"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_routes={
        "app.tasks.ingest_task.*": {"queue": "ingestion"},
    },
    result_expires=86400,  # 1 day
)
