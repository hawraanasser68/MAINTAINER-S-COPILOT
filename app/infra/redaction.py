import re

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9\-_]{20,}")),
    ("openai_key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("github_pat_classic", re.compile(r"ghp_[A-Za-z0-9]{36}")),
    ("github_pat_fine", re.compile(r"github_pat_[A-Za-z0-9_]{82}")),
    ("aws_access_key", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*")),
    ("email", re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")),
]

_PLACEHOLDER = "[REDACTED]"


def redact(text: str) -> str:
    """Replace all sensitive patterns with [REDACTED]. Must be called before
    any log write, span attribute assignment, or memory write."""
    for _, pattern in _PATTERNS:
        text = pattern.sub(_PLACEHOLDER, text)
    return text


def redact_dict(data: dict) -> dict:
    """Recursively redact all string values in a dict (for span attributes)."""
    result = {}
    for k, v in data.items():
        if isinstance(v, str):
            result[k] = redact(v)
        elif isinstance(v, dict):
            result[k] = redact_dict(v)
        else:
            result[k] = v
    return result
