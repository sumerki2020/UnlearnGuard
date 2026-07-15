"""Persist audit outputs to Nebius object storage (S3-compatible).

Job disks are ephemeral and job logs expire, so results/cards must be uploaded
before the container exits. Credentials come from the environment (never
committed): S3_ENDPOINT, S3_ACCESS_KEY_ID, S3_SECRET_ACCESS_KEY, S3_BUCKET.

CLI:
  python -m src.artifacts upload <local_dir> <prefix>
  python -m src.artifacts download <prefix> <local_dir>
"""

import os
import sys
from pathlib import Path


def _client():
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=os.environ["S3_ENDPOINT"],
        aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
    )


def _configured():
    return all(os.environ.get(k) for k in
               ("S3_ENDPOINT", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY", "S3_BUCKET"))


def upload_dir(local_dir, prefix, bucket=None):
    if not _configured():
        print("[artifacts] S3 env not set — skipping upload")
        return 0
    bucket = bucket or os.environ["S3_BUCKET"]
    c, n = _client(), 0
    for p in sorted(Path(local_dir).rglob("*")):
        if p.is_file():
            key = f"{prefix}/{p.relative_to(local_dir)}"
            c.upload_file(str(p), bucket, key)
            n += 1
    print(f"[artifacts] uploaded {n} files to s3://{bucket}/{prefix}")
    return n


def download_dir(prefix, local_dir, bucket=None):
    bucket = bucket or os.environ["S3_BUCKET"]
    c = _client()
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    paginator = c.get_paginator("list_objects_v2")
    n = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):].lstrip("/")
            dest = Path(local_dir) / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            c.download_file(bucket, obj["Key"], str(dest))
            n += 1
    print(f"[artifacts] downloaded {n} files from s3://{bucket}/{prefix} -> {local_dir}")
    return n


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "upload":
        upload_dir(sys.argv[2], sys.argv[3])
    elif cmd == "download":
        download_dir(sys.argv[2], sys.argv[3])
    else:
        raise SystemExit(f"unknown command {cmd}")
