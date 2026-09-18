"""
Object storage client — any S3-compatible endpoint via boto3.
Supabase Storage and Cloudflare R2 are the documented options; the R2_*
env names are historical.
All media goes here — never to local filesystem in production.
"""

from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import BinaryIO

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

log = logging.getLogger(__name__)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = boto3.client(
            "s3",
            endpoint_url=os.environ["R2_ENDPOINT"],
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            # Path-style addressing: required by Supabase Storage, accepted by R2.
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                retries={"max_attempts": 4, "mode": "standard"},
                connect_timeout=10,
                read_timeout=60,
                tcp_keepalive=True,
            ),
            region_name=os.getenv("R2_REGION", "auto"),
        )
    return _client


BUCKET = os.getenv("R2_BUCKET", "clippy")


def upload_file(local_path: str, r2_key: str, content_type: str = "application/octet-stream") -> str:
    """Upload local file to R2. Returns the R2 key."""
    client = _get_client()
    client.upload_file(
        local_path,
        BUCKET,
        r2_key,
        ExtraArgs={"ContentType": content_type},
    )
    log.info("Uploaded %s → r2://%s/%s", local_path, BUCKET, r2_key)
    return r2_key


def upload_bytes(data: bytes, r2_key: str, content_type: str = "application/octet-stream") -> str:
    """Upload bytes to R2. Returns the R2 key."""
    import io
    client = _get_client()
    client.upload_fileobj(
        io.BytesIO(data),
        BUCKET,
        r2_key,
        ExtraArgs={"ContentType": content_type},
    )
    log.info("Uploaded bytes → r2://%s/%s", BUCKET, r2_key)
    return r2_key


def download_file(r2_key: str, local_path: str) -> None:
    """Download R2 object to local path."""
    client = _get_client()
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    client.download_file(BUCKET, r2_key, local_path)
    log.info("Downloaded r2://%s/%s → %s", BUCKET, r2_key, local_path)


def get_presigned_url(r2_key: str, expires_in: int = 3600) -> str:
    """Generate a presigned GET URL (1 hour default)."""
    client = _get_client()
    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": BUCKET, "Key": r2_key},
        ExpiresIn=expires_in,
    )


def key_exists(r2_key: str) -> bool:
    """Check whether an object exists without hiding transient storage failures."""
    try:
        _get_client().head_object(Bucket=BUCKET, Key=r2_key)
        return True
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if code in {"404", "NoSuchKey", "NotFound"} or status == 404:
            return False
        raise


def delete_keys(keys: list[str]) -> None:
    """Delete multiple R2 objects in batches of 1000. Noop on empty list."""
    if not keys:
        return
    client = _get_client()
    for i in range(0, len(keys), 1000):
        batch = [{"Key": k} for k in keys[i : i + 1000]]
        client.delete_objects(Bucket=BUCKET, Delete={"Objects": batch})
        log.info("Deleted %d R2 objects", len(batch))


def public_url(r2_key: str) -> str:
    """
    Return the public URL for an R2 object (requires bucket to be public
    or Cloudflare CDN configured). Falls back to a 7-day presigned URL.
    """
    public_domain = os.getenv("R2_PUBLIC_DOMAIN")
    if public_domain:
        return f"https://{public_domain}/{r2_key}"
    return get_presigned_url(r2_key, expires_in=604800)  # 7 days (R2 max)
