import os
from functools import lru_cache

import hvac

from app.domain.exceptions import InfrastructureError

_VAULT_ADDR = os.environ.get("VAULT_ADDR", "http://vault:8200")
_VAULT_TOKEN = os.environ.get("VAULT_ROOT_TOKEN", "")


@lru_cache(maxsize=1)
def _client() -> hvac.Client:
    client = hvac.Client(url=_VAULT_ADDR, token=_VAULT_TOKEN)
    if not client.is_authenticated():
        raise InfrastructureError("vault", "not authenticated")
    return client


def get_secret(path: str, key: str = "value") -> str:
    """Read a KV-v2 secret from Vault. Path is relative to secret/."""
    try:
        client = _client()
        response = client.secrets.kv.v2.read_secret_version(path=path, mount_point="secret")
        return response["data"]["data"][key]
    except InfrastructureError:
        raise
    except Exception as exc:
        raise InfrastructureError("vault", str(exc)) from exc


def check_connectivity() -> bool:
    """Return True if Vault is reachable and authenticated."""
    try:
        _client()
        return True
    except Exception:
        return False


def seed_dev_secrets() -> None:
    """Seed development secrets into Vault (dev mode only). Called at compose startup."""
    try:
        client = _client()
        secrets = {
            "db/password": {"value": "postgres"},
            "redis/password": {"value": ""},
            "minio/access_key": {"value": "minioadmin"},
            "minio/secret_key": {"value": "minioadmin"},
            "jwt/signing_key": {"value": "dev-jwt-signing-key-change-in-prod"},
            "github/token": {"value": os.environ.get("GITHUB_TOKEN", "")},
            # These keys are NOT seeded — must be put into Vault manually before boot:
            #   docker exec -e VAULT_ADDR=http://127.0.0.1:8200 -e VAULT_TOKEN=dev-root-token \
            #     vault vault kv put secret/llm/groq_api_key value=gsk_...
            #   docker exec -e VAULT_ADDR=http://127.0.0.1:8200 -e VAULT_TOKEN=dev-root-token \
            #     vault vault kv put secret/llm/anthropic_api_key value=sk-ant-...
            "tracing/endpoint": {"value": "http://jaeger:4317"},
        }
        for path, data in secrets.items():
            client.secrets.kv.v2.create_or_update_secret(path=path, secret=data, mount_point="secret")
    except Exception:
        pass  # Dev seed is best-effort
