import uuid

from fastapi import Request
from fastapi.responses import JSONResponse

from app.domain.exceptions import (
    AppError,
    InfrastructureError,
    NotFoundError,
    PermissionDenied,
    ToolFailure,
    ValidationError,
)

_STATUS_MAP: dict[type[AppError], int] = {
    NotFoundError: 404,
    PermissionDenied: 403,
    ToolFailure: 502,
    InfrastructureError: 503,
    ValidationError: 422,
}


def _error_response(request: Request, status: int, code: str, message: str) -> JSONResponse:
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=status,
        content={"error": {"code": code, "message": message, "request_id": request_id}},
    )


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    status = _STATUS_MAP.get(type(exc), 500)
    return _error_response(request, status, exc.code, exc.message)


async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    import logging

    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    trace_id = getattr(request.state, "trace_id", "unknown")
    logging.getLogger(__name__).error(
        "Unhandled exception",
        extra={"request_id": request_id, "trace_id": trace_id},
        exc_info=exc,
    )
    return _error_response(request, 500, "INTERNAL_ERROR", "An unexpected error occurred.")


def register_exception_handlers(app) -> None:  # type: ignore[no-untyped-def]
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(Exception, unhandled_error_handler)
