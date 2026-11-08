import os
import boto3
from botocore.config import Config

s3 = boto3.client(
    "s3",
    endpoint_url=os.environ.get("S3_ENDPOINT", "http://127.0.0.1:8333"),
    aws_access_key_id="dummy",
    aws_secret_access_key="dummy",
    region_name="us-east-1",
    config=Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"},
    ),
)

expected = {
    "datasets": {"commands_pool.jsonl", "schema.json"},
    "models": {
        f"{version}/{filename}"
        for version in ("v1", "v2")
        for filename in (
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "training_args.bin",
            "training_metadata.json",
        )
    },
}

all_ok = True

for bucket, required_keys in expected.items():
    paginator = s3.get_paginator("list_objects_v2")
    actual_keys = {
        obj["Key"]
        for page in paginator.paginate(Bucket=bucket)
        for obj in page.get("Contents", [])
    }

    missing = required_keys - actual_keys

    print(f"\nBucket: {bucket}")
    print(f"Expected objects: {len(required_keys)}")
    print(f"Actual objects: {len(actual_keys)}")
    print(f"Missing objects: {len(missing)}")

    if missing:
        all_ok = False
        for key in sorted(missing):
            print(f"  MISSING: {key}")
    else:
        print("All required objects are present.")

if not all_ok:
    raise SystemExit("VERIFICATION FAILED")

print("\nSUCCESS: All 18 required objects are present in SeaweedFS.")
