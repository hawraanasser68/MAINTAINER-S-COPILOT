class AppError(Exception):
    """Base domain exception. All domain errors inherit from this."""

    def __init__(self, message: str, code: str | None = None) -> None:
        self.message = message
        self.code = code or self.__class__.__name__
        super().__init__(message)


class NotFoundError(AppError):
    def __init__(self, resource: str, identifier: str | int) -> None:
        super().__init__(f"{resource} '{identifier}' not found", "NOT_FOUND")


class PermissionDenied(AppError):
    def __init__(self, action: str) -> None:
        super().__init__(f"Permission denied: {action}", "PERMISSION_DENIED")


class ToolFailure(AppError):
    def __init__(self, tool: str, reason: str) -> None:
        super().__init__(f"Tool '{tool}' failed: {reason}", "TOOL_FAILURE")
        self.tool = tool


class InfrastructureError(AppError):
    def __init__(self, component: str, reason: str = "") -> None:
        msg = f"Infrastructure error [{component}]"
        if reason:
            msg += f": {reason}"
        super().__init__(msg, "INFRASTRUCTURE_ERROR")
        self.component = component


class ValidationError(AppError):
    def __init__(self, field: str, reason: str) -> None:
        super().__init__(f"Validation failed for '{field}': {reason}", "VALIDATION_ERROR")
