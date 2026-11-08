"""CS3081 Task 1: reliable, resumable, sharded Phi-3.5 labelling worker."""

from __future__ import annotations

import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
MODEL = "phi3.5"
TEMPERATURE = 0
SEED = 42
MAX_ATTEMPTS = int(os.environ.get("MAX_ATTEMPTS", "5"))
BACKOFF_BASE = float(os.environ.get("BACKOFF_BASE", "2"))
BACKOFF_MAX = float(os.environ.get("BACKOFF_MAX", "30"))
REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "300"))
UPLOAD_EVERY = int(os.environ.get("UPLOAD_EVERY", "25"))
S3_TEACHER_PREFIX = os.environ.get("S3_TEACHER_PREFIX", "teacher/structured-v1").strip("/")


def load_prompt_template(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def load_json_schema(path: str) -> dict:
    """Load the professor-provided Ollama-compatible JSON Schema."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    required_keys = {"intent", "date", "time", "place"}
    if set(document.get("keys", [])) != required_keys:
        raise ValueError("schema.json must define exactly intent/date/time/place")
    if not document.get("intents"):
        raise ValueError("schema.json does not define the permitted intents")
    schema = document.get("json_schema")
    if not isinstance(schema, dict):
        raise ValueError("schema.json is missing its json_schema object")
    return schema


def call_teacher(prompt_template: str, command_text: str, json_schema: dict) -> str:
    """Generate one response; return the teacher's raw response unchanged."""
    prompt = prompt_template.replace("{command}", command_text)
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": json_schema,
        "options": {"temperature": TEMPERATURE, "seed": SEED},
    }
    body = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        request = urllib.request.Request(
            f"{OLLAMA_HOST}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                envelope = json.loads(response.read().decode("utf-8"))
            raw = envelope.get("response")
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError("Ollama returned an empty or non-string response")
            return raw
        except urllib.error.HTTPError as exc:
            # Retry throttling/server errors; fail immediately on other 4xx errors.
            if exc.code not in (408, 429) and not 500 <= exc.code <= 599:
                raise
            last_error = exc
        except (urllib.error.URLError, TimeoutError, socket.timeout,
                ConnectionError, OSError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc

        if attempt < MAX_ATTEMPTS:
            delay = min(BACKOFF_BASE * (2 ** (attempt - 1)), BACKOFF_MAX)
            print(
                f"[teacher] attempt {attempt}/{MAX_ATTEMPTS} failed: {last_error}; "
                f"retrying in {delay:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError(
        f"Teacher request failed after {MAX_ATTEMPTS} attempts: {last_error}"
    )


def get_s3_client():
    endpoint = os.environ.get("S3_ENDPOINT")
    bucket = os.environ.get("S3_BUCKET")
    if not endpoint and not bucket:
        return None, None
    if not endpoint or not bucket:
        raise RuntimeError("S3_ENDPOINT and S3_BUCKET must be set together")

    access_key = os.environ.get("S3_ACCESS_KEY")
    secret_key = os.environ.get("S3_SECRET_KEY")
    if not access_key or not secret_key:
        raise RuntimeError("S3 credentials must be supplied through environment/Secret")

    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=Config(s3={"addressing_style": "path"}),
    )
    return client, bucket


def checkpoint_key(shard_index: int) -> str:
    return f"{S3_TEACHER_PREFIX}/shard-{shard_index}.jsonl"


def maybe_download_commands(commands_path: str, s3, bucket: str | None) -> None:
    if Path(commands_path).is_file():
        return
    if s3 is None or bucket is None:
        raise FileNotFoundError(
            f"{commands_path} not found and no S3 source is configured"
        )
    key = os.environ.get("S3_COMMANDS_KEY", "commands_pool.jsonl")
    Path(commands_path).parent.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, key, commands_path)
    print(f"Downloaded s3://{bucket}/{key} -> {commands_path}", flush=True)


def download_checkpoint_if_present(output_path: str, shard_index: int, s3, bucket) -> None:
    if s3 is None:
        return
    from botocore.exceptions import ClientError

    key = checkpoint_key(shard_index)
    try:
        s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return
        raise
    s3.download_file(bucket, key, output_path)
    print(f"[shard {shard_index}] resumed from s3://{bucket}/{key}", flush=True)


def upload_output(output_path: str, shard_index: int, s3, bucket) -> None:
    if s3 is None:
        return
    key = checkpoint_key(shard_index)
    s3.upload_file(output_path, bucket, key)
    print(f"[shard {shard_index}] uploaded s3://{bucket}/{key}", flush=True)


def load_existing_ids(output_path: str) -> set[str]:
    """Read the local/S3 checkpoint, rejecting corrupt or duplicate JSONL rows."""
    done: set[str] = set()
    path = Path(output_path)
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                command_id = record["id"]
                if not isinstance(command_id, str):
                    raise ValueError("id is not a string")
                if not isinstance(record["text"], str):
                    raise ValueError("text is not a string")
                if not isinstance(record["teacher_raw"], str):
                    raise ValueError("teacher_raw is not a string")
            except Exception as exc:
                raise ValueError(f"Invalid checkpoint {output_path}:{line_no}: {exc}") from exc
            if command_id in done:
                raise ValueError(f"Duplicate checkpoint ID {command_id} in {output_path}")
            done.add(command_id)
    return done


def read_shard_records(commands_path: str, shard_index: int, shard_count: int) -> list[dict]:
    """Shard on the zero-based physical line number, as required by the brief."""
    records = []
    with open(commands_path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if not line.strip():
                continue
            if line_number % shard_count != shard_index:
                continue
            record = json.loads(line)
            if not isinstance(record.get("id"), str) or not isinstance(record.get("text"), str):
                raise ValueError(f"Invalid input record at physical line {line_number + 1}")
            records.append(record)
    ids = [record["id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate input IDs assigned to shard {shard_index}")
    return records


def main() -> None:
    shard_index = int(os.environ["SHARD_INDEX"])
    shard_count = int(os.environ["SHARD_COUNT"])
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("SHARD_INDEX must be in [0, SHARD_COUNT)")
    if MAX_ATTEMPTS < 1 or UPLOAD_EVERY < 1 or BACKOFF_BASE < 0 or BACKOFF_MAX < 0:
        raise ValueError("Retry and upload configuration values are invalid")

    commands_path = os.environ.get("COMMANDS_PATH", "data/commands_pool.jsonl")
    prompt_path = os.environ.get("PROMPT_PATH", "prompts/teacher_prompt.txt")
    schema_path = os.environ.get("SCHEMA_PATH", "data/schema.json")
    output_path = os.environ.get(
        "OUTPUT_PATH", f"datasets/teacher/shard-{shard_index}.jsonl"
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    s3, bucket = get_s3_client()
    maybe_download_commands(commands_path, s3, bucket)
    if not Path(output_path).exists():
        download_checkpoint_if_present(output_path, shard_index, s3, bucket)

    prompt_template = load_prompt_template(prompt_path)
    json_schema = load_json_schema(schema_path)
    records = read_shard_records(commands_path, shard_index, shard_count)
    expected_ids = {record["id"] for record in records}
    done = load_existing_ids(output_path)

    unexpected = done - expected_ids
    if unexpected:
        raise ValueError(
            f"Checkpoint has IDs outside shard {shard_index}: {sorted(unexpected)[:10]}"
        )

    print(
        f"[shard {shard_index}] expected={len(expected_ids)} "
        f"already_done={len(done)} remaining={len(expected_ids - done)}",
        flush=True,
    )

    newly_labelled = 0
    failed_ids = []
    with open(output_path, "a", encoding="utf-8") as out:
        for record in records:
            command_id = record["id"]
            if command_id in done:
                continue
            try:
                teacher_raw = call_teacher(prompt_template, record["text"], json_schema)
            except Exception as exc:
                failed_ids.append(command_id)
                print(
                    f"[shard {shard_index}] FAILED {command_id}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                continue

            # Preserve the teacher response exactly as returned by Ollama.
            row = {
                "id": command_id,
                "text": record["text"],
                "teacher_raw": teacher_raw,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            done.add(command_id)
            newly_labelled += 1
            print(
                f"[shard {shard_index}] labelled {command_id} "
                f"({len(done)}/{len(expected_ids)})",
                flush=True,
            )

            # Upload is deliberately outside the generation try/except: storage
            # errors must fail the Job rather than be misreported as label errors.
            if newly_labelled % UPLOAD_EVERY == 0:
                upload_output(output_path, shard_index, s3, bucket)

    # Upload the final partial batch before declaring success/failure.
    upload_output(output_path, shard_index, s3, bucket)
    missing = expected_ids - done
    print(
        f"[shard {shard_index}] SUMMARY expected={len(expected_ids)} "
        f"completed={len(done)} newly_labelled={newly_labelled} "
        f"failed={len(failed_ids)} missing={len(missing)}",
        flush=True,
    )
    if failed_ids or missing:
        print(
            f"[shard {shard_index}] incomplete IDs: {sorted(missing)[:30]}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)
    print(f"[shard {shard_index}] SUCCESS: all assigned commands labelled", flush=True)


if __name__ == "__main__":
    main()
