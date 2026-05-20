# Quickstart: Phase 1 — Foundations

## Prerequisites

- Docker + Docker Compose installed
- Python 3.11 (for running scripts locally)
- A GitHub personal access token (for dataset fetch)

## Start The Stack

```bash
git clone <repo-url>
cd maintainers-copilot
cp .env.example .env
# Edit .env: set VAULT_ROOT_TOKEN to any string (dev mode)
```

```bash
docker-compose up --build
```

Wait for `migrate` to exit 0, then `api` starts. All services healthy in ~2 minutes.

## Verify Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "vault": "reachable",
  "db": "reachable",
  "redis": "reachable",
  "tracing": "configured"
}
```

## Fetch And Split The Dataset

```bash
# Inside the container or in a local venv with dependencies installed
python scripts/fetch_issues.py --repo owner/repo --output data/raw_issues.json
python scripts/split_dataset.py --input data/raw_issues.json --output-dir data/splits/
```

Outputs:
- `data/splits/train.jsonl`
- `data/splits/val.jsonl`
- `data/splits/test.jsonl`
- `data/splits/split_stats.json` (label counts per split)

## View Traces

Open Jaeger UI: http://localhost:16686

Select service `maintainers-copilot-api` and inspect spans from the `/health` call.

## Run Tests

```bash
docker-compose run --rm api pytest tests/
```

All three test files must pass: `test_health.py`, `test_redaction.py`, `test_splits.py`.

## Phase 1 Done When

- [ ] `docker-compose up` produces a fully running stack
- [ ] `GET /health` returns 200 with all dependencies confirmed
- [ ] Dataset splits exist with correct stratification and temporal ordering
- [ ] At least one trace visible in Jaeger after any API call
- [ ] `grep -ri 'sk-' app/` returns zero matches outside `app/infra/vault.py`
- [ ] All three tests pass
