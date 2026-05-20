# Runbook

## Start The Stack

```bash
cp .env.example .env
# VAULT_ROOT_TOKEN is already set to "dev-root-token" — change if needed

# Step 1: start Vault first so you can seed secrets
docker-compose up -d vault

# Step 2: put your Groq API key directly into Vault (never goes in .env)
docker-compose exec vault vault kv put secret/llm/groq_api_key value=gsk_YOUR_KEY_HERE

# Step 3: start everything else
docker-compose up --build
```

Wait for `migrate` to exit 0 before the api is ready (~2 min on first build).

## Verify Health

```bash
curl http://localhost:8000/health
```

## Run Migrations Manually

```bash
docker-compose run --rm migrate
# or locally:
alembic upgrade head
```

## Fetch Dataset

```bash
# Requires GITHUB_TOKEN in Vault at secret/github/token
python scripts/fetch_issues.py --repo huggingface/transformers --output data/raw_issues.jsonl
```

## Split Dataset

```bash
python scripts/split_dataset.py --input data/raw_issues.jsonl --output-dir data/splits/
```

## Run Tests

```bash
docker-compose run --rm api pytest tests/ -v
```

## View Traces

Open http://localhost:16686 — select service `maintainers-copilot-api`.

## View MinIO

Open http://localhost:9001 — login with credentials from Vault (`secret/minio`).

## Access Vault UI

Open http://localhost:8200 — use `VAULT_ROOT_TOKEN` from `.env`.
