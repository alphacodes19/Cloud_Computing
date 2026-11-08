import hashlib
import os
from pathlib import Path

import boto3
from botocore.config import Config

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "models"
VERSIONS = ("v1", "v2")

endpoint = os.environ.get("S3_ENDPOINT", "http://127.0.0.1:8333")
access_key = os.environ.get("AWS_ACCESS_KEY_ID", "dummy")
secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "dummy")

s3 = boto3.client(
    "s3",
    endpoint_url=endpoint,
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
    config=Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"},
    ),
)

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def sha256_remote(bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    digest = hashlib.sha256()
    size = 0
    body = response["Body"]
    try:
        for chunk in body.iter_chunks(chunk_size=8 * 1024 * 1024):
            if chunk:
                digest.update(chunk)
                size += len(chunk)
    finally:
        body.close()
    return digest.hexdigest(), size

print(f"SeaweedFS endpoint: {endpoint}")
print(f"Checking bucket: {BUCKET}")
s3.head_bucket(Bucket=BUCKET)

manifest = ROOT / "results" / "model-artifact-checksums.txt"
manifest.parent.mkdir(parents=True, exist_ok=True)

total_files = 0
total_bytes = 0
failures = []

with manifest.open("w", encoding="utf-8") as report:
    report.write("CS3081 model artifact SHA-256 verification\n")
    report.write(f"Bucket: {BUCKET}\n\n")

    for version in VERSIONS:
        model_dir = ROOT / "models" / version
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Missing model directory: {model_dir}")

        files = sorted(p for p in model_dir.rglob("*") if p.is_file())
        if not files:
            raise RuntimeError(f"No model files found in {model_dir}")

        print(f"\nUploading models/{version}/ ({len(files)} files)")

        for path in files:
            relative = path.relative_to(model_dir).as_posix()
            key = f"{version}/{relative}"
            local_hash = sha256_file(path)
            local_size = path.stat().st_size

            print(f"  Uploading {key} ({local_size:,} bytes)")
            s3.upload_file(
                str(path),
                BUCKET,
                key,
                ExtraArgs={"Metadata": {"sha256": local_hash}},
            )

            remote_hash, remote_size = sha256_remote(BUCKET, key)
            matches = (
                local_hash == remote_hash
                and local_size == remote_size
            )

            status = "PASS" if matches else "FAIL"
            print(f"  {status}: {key}")

            report.write(
                f"{status}  {key}  size={local_size}  "
                f"sha256={local_hash}\n"
            )

            total_files += 1
            total_bytes += local_size

            if not matches:
                failures.append(key)

    report.write(f"\nFiles verified: {total_files}\n")
    report.write(f"Bytes uploaded: {total_bytes}\n")
    report.write(f"Failures: {len(failures)}\n")

print("\n--- Upload summary ---")
print(f"Files processed: {total_files}")
print(f"Total local bytes: {total_bytes:,}")
print(f"Checksum failures: {len(failures)}")
print(f"Manifest: {manifest}")

if failures:
    print("Failed objects:")
    for key in failures:
        print(f"  {key}")
    raise SystemExit(1)

print("SUCCESS: All uploaded model files passed SHA-256 verification.")
