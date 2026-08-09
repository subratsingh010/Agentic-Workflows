from __future__ import annotations

from app.core.config import Settings

_TRACING_CONFIGURED = False


def setup_observability(app, settings: Settings) -> None:
    _setup_metrics(app)
    _setup_tracing(app, settings)


def _setup_metrics(app) -> None:
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
    except Exception:
        return
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")


def _setup_tracing(app, settings: Settings) -> None:
    global _TRACING_CONFIGURED
    if _TRACING_CONFIGURED:
        return
    endpoints = [
        endpoint
        for endpoint in (settings.otel_exporter_otlp_endpoint, settings.phoenix_collector_endpoint)
        if endpoint
    ]
    if not endpoints:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except Exception:
        return

    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": settings.app_name,
                "service.environment": settings.app_env,
                "telemetry.sdk.language": "python",
            }
        )
    )
    try:
        for endpoint in endpoints:
            provider.add_span_processor(BatchSpanProcessor(_span_exporter(endpoint)))
        trace.set_tracer_provider(provider)
    except Exception:
        return
    FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
    HTTPXClientInstrumentor().instrument(tracer_provider=provider)
    _TRACING_CONFIGURED = True


def _span_exporter(endpoint: str):
    if endpoint.endswith("/v1/traces"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter as HTTPSpanExporter,
        )

        return HTTPSpanExporter(endpoint=endpoint)
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as GRPCSpanExporter,
    )

    return GRPCSpanExporter(endpoint=endpoint)
