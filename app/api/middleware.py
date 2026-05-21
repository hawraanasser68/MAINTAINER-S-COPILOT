import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.infra.tracing import get_current_trace_id

logger = structlog.get_logger(__name__)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attaches a UUID request_id to every request and injects it into logs and response headers."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        trace_id = get_current_trace_id()
        request.state.trace_id = trace_id
        with structlog.contextvars.bound_contextvars(request_id=request_id, trace_id=trace_id):
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Trace-ID"] = trace_id
            return response
