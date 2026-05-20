from contextlib import contextmanager
from typing import Any, Generator

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from app.infra.redaction import redact_dict

_tracer: trace.Tracer | None = None


def setup_tracing(service_name: str, otlp_endpoint: str) -> None:
    global _tracer
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(service_name)


def get_tracer() -> trace.Tracer:
    if _tracer is None:
        # Fallback to no-op tracer during tests
        return trace.get_tracer("maintainers-copilot")
    return _tracer


def get_current_trace_id() -> str:
    span = trace.get_current_span()
    ctx = span.get_span_context()
    if ctx and ctx.is_valid:
        return format(ctx.trace_id, "032x")
    return "no-trace"


@contextmanager
def create_span(
    name: str, attributes: dict[str, Any] | None = None
) -> Generator[trace.Span, None, None]:
    """Context manager that creates a span with redacted attributes."""
    tracer = get_tracer()
    safe_attrs = redact_dict(attributes or {})
    with tracer.start_as_current_span(name, attributes=safe_attrs) as span:
        yield span
