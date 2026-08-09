from app.core.config import Settings


def setup_observability(app, settings: Settings) -> None:
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
    except Exception:
        return
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    # OTEL and Phoenix exporters are configured here in real deployments. The
    # scaffold keeps this side-effect light so tests and local startup remain simple.
