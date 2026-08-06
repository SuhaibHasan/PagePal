from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

SERVICE_NAME = "prodsupportbuddy-api"

_configured = False


def configure_tracing() -> None:
    # Idempotent: FastAPI's module can be imported multiple times (e.g. by
    # tests), and re-configuring the global TracerProvider is a no-op anyway.
    global _configured
    if _configured:
        return

    # ConsoleSpanExporter keeps this fully local by default, matching the rest
    # of the project; swap in an OTLP exporter (with BatchSpanProcessor, which
    # needs a real network sink to batch for) to ship spans to a real collector
    # in production. SimpleSpanProcessor exports synchronously with no
    # background thread, which is what a Console sink actually wants.
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}))
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _configured = True


tracer = trace.get_tracer(SERVICE_NAME)
