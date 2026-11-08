"""Initialize SeaweedFS buckets and upload CS3081 project artifacts."""

import hashlib
import os
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = os.environ.get("S3_ENDPOINT", "http://127.0.0.1:18333")
ACCESS_KEY = os.environ.get("S3_ACCESS_KEY", "dummy")
SECRET_KEY = os.environ.get("S3_SECRET_KEY", "dummy")

s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=ACCESS_KEY,
    aws_secret_access_key=SECRET_KEY,
    region_name="us-east-1",
    config=Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"},
    ),
)


def ensure_bucket(name):
    try:
        s3.head_bucket(Bucket=name)
        print(f"Bucket already exists: {name}")
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        status = exc.response.get("ResponseMetadata", {}).get(
            "HTTPStatusCode"
        )
        if str(code) not in {"404", "NoSuchBucket", "NotFound"} and status != 404:
            raise
        s3.create_bucket(Bucket=name)
        print(f"Created bucket: {name}")


def upload_file(bucket, key, path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    s3.upload_file(
        str(path),
        bucket,
        key,
        ExtraArgs={"Metadata": {"sha256": digest}},
    )
    print(f"Uploaded s3://{bucket}/{key} ({path.stat().st_size:,} bytes)")


def main():
    print(f"SeaweedFS endpoint: {ENDPOINT}")

    ensure_bucket("datasets")
    ensure_bucket("models")

    # Upload the command pool and schema. The validation set is deliberately
    # not uploaded as training data.
    upload_file(
        "datasets",
        "commands_pool.jsonl",
        ROOT / "data" / "commands_pool.jsonl",
    )
    upload_file(
        "datasets",
        "schema.json",
        ROOT / "data" / "schema.json",
    )

    # Upload both trained versions, preserving their directory structure.
    for version in ("v1", "v2"):
        model_dir = ROOT / "models" / version
        if not (model_dir / "config.json").is_file():
            raise FileNotFoundError(f"Missing {model_dir}/config.json")
        if not (model_dir / "model.safetensors").is_file():
            raise FileNotFoundError(f"Missing {model_dir}/model.safetensors")

        files = sorted(path for path in model_dir.rglob("*") if path.is_file())
        for path in files:
            relative = path.relative_to(model_dir).as_posix()
            upload_file("models", f"{version}/{relative}", path)

    print("\nSUCCESS: SeaweedFS buckets and project artifacts initialized.")


if __name__ == "__main__":
    main()
