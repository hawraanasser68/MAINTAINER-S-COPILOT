from io import BytesIO

import boto3
from botocore.exceptions import ClientError, EndpointResolutionError

from app.domain.exceptions import InfrastructureError, NotFoundError

_client = None
_endpoint: str = ""


def init_minio(endpoint: str, access_key: str, secret_key: str) -> None:
    global _client, _endpoint
    _endpoint = endpoint
    _client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="us-east-1",
    )


def _get():  # type: ignore[no-untyped-def]
    if _client is None:
        raise InfrastructureError("minio", "not initialised — call init_minio() first")
    return _client


def ensure_bucket(bucket: str) -> None:
    client = _get()
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError:
        client.create_bucket(Bucket=bucket)


def upload(bucket: str, key: str, data: bytes | str) -> None:
    ensure_bucket(bucket)
    body = data.encode() if isinstance(data, str) else data
    _get().put_object(Bucket=bucket, Key=key, Body=body)


def download(bucket: str, key: str) -> bytes:
    try:
        response = _get().get_object(Bucket=bucket, Key=key)
        return response["Body"].read()
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "NoSuchKey":
            raise NotFoundError("MinIO object", key) from exc
        raise InfrastructureError("minio", str(exc)) from exc


def list_objects(bucket: str, prefix: str = "") -> list[str]:
    ensure_bucket(bucket)
    response = _get().list_objects_v2(Bucket=bucket, Prefix=prefix)
    return [obj["Key"] for obj in response.get("Contents", [])]


def check_connectivity() -> bool:
    if _client is None:
        return False
    try:
        _get().list_buckets()
        return True
    except Exception:
        return False
