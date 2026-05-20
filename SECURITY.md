# Security

## Redaction Patterns

All patterns applied by `app/infra/redaction.py` before any log line, span attribute, or memory write leaves the service boundary.

| Pattern | Regex | Rationale |
|---|---|---|
| OpenAI / Anthropic API keys | `sk-[A-Za-z0-9]{20,}` | API keys from LLM providers appear in issue text pasted by users |
| GitHub PATs (classic) | `ghp_[A-Za-z0-9]{36}` | Maintainers paste stack traces containing tokens |
| GitHub PATs (fine-grained) | `github_pat_[A-Za-z0-9_]{82}` | Same risk |
| Email addresses | `[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}` | PII — appears in issue reporter fields |
| Generic bearer tokens | `Bearer\s+[A-Za-z0-9\-._~+/]+=*` | Auth headers that may appear in curl examples in issues |
| AWS access keys | `AKIA[0-9A-Z]{16}` | Cloud credentials pasted in bug reports |

## Where Redaction Is Applied

- Every `structlog` log line — via a structlog processor added at logger setup
- Every OpenTelemetry span attribute — enforced inside `create_span()` in `app/infra/tracing.py`
- Every long-term memory write — called in the memory service before persisting

## Tested By

`tests/test_redaction.py` asserts that a string containing `sk-fake123456789012345` never appears unredacted in log output, span attributes, or memory writes.
