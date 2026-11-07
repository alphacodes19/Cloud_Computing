"""Task 1 worker: labels a shard of commands_pool.jsonl with the teacher model.

Reads SHARD_INDEX and SHARD_COUNT from the environment. Takes the commands
whose line number (counting from 0 in commands_pool.jsonl) leaves remainder
SHARD_INDEX when divided by SHARD_COUNT, sends each to the teacher (Ollama,
model phi3.5) using prompts/teacher_prompt.txt, and appends one JSON line per
command to its own shard output file.

Environment variables:
  SHARD_INDEX     this worker's shard index, starting at 0 (required)
  SHARD_COUNT     total number of shards running (required)
  OLLAMA_HOST     base URL of the Ollama server (default: http://localhost:11434)
  COMMANDS_PATH   path to commands_pool.jsonl (default: data/commands_pool.jsonl)
  PROMPT_PATH     path to the prompt template (default: prompts/teacher_prompt.txt)
  OUTPUT_PATH     where to write this shard's output
                  (default: datasets/teacher/shard-{SHARD_INDEX}.jsonl)
  S3_ENDPOINT, S3_BUCKET, S3_ACCESS_KEY, S3_SECRET_KEY
                  if set, the shard output is also uploaded to SeaweedFS via boto3

This script is idempotent on restart: it skips command ids already present
in OUTPUT_PATH, so a killed/restarted worker resumes instead of relabelling.
"""
import json
import os
import sys
import time
import urllib.request

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = "phi3.5"
TEMPERATURE = 0
SEED = 42


def load_prompt_template(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def call_teacher(prompt_template, command_text):
    prompt = prompt_template.replace("{command}", command_text)
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": TEMPERATURE, "seed": SEED},
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["response"]


def already_done(output_path):
    done = set()
    if os.path.exists(output_path):
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    continue
    return done


def main():
    shard_index = int(os.environ["SHARD_INDEX"])
    shard_count = int(os.environ["SHARD_COUNT"])
    commands_path = os.environ.get("COMMANDS_PATH", "data/commands_pool.jsonl")
    prompt_path = os.environ.get("PROMPT_PATH", "prompts/teacher_prompt.txt")
    output_path = os.environ.get("OUTPUT_PATH", f"datasets/teacher/shard-{shard_index}.jsonl")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prompt_template = load_prompt_template(prompt_path)
    done = already_done(output_path)

    with open(commands_path, encoding="utf-8") as f, open(output_path, "a", encoding="utf-8") as out:
        for line_number, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if line_number % shard_count != shard_index:
                continue
            record = json.loads(line)
            if record["id"] in done:
                continue
            try:
                teacher_raw = call_teacher(prompt_template, record["text"])
            except Exception as exc:  # keep going; note failures, don't crash the shard
                print(f"[shard {shard_index}] FAILED {record['id']}: {exc}", file=sys.stderr)
                continue
            out.write(json.dumps({
                "id": record["id"],
                "text": record["text"],
                "teacher_raw": teacher_raw,
            }) + "\n")
            out.flush()
            print(f"[shard {shard_index}] labelled {record['id']}")

    # Optional: push the shard file to SeaweedFS (S3-compatible) if configured.
    endpoint = os.environ.get("S3_ENDPOINT")
    bucket = os.environ.get("S3_BUCKET")
    if endpoint and bucket:
        import boto3
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
            aws_secret_access_key=os.environ.get("S3_SECRET_KEY"),
        )
        key = f"teacher/shard-{shard_index}.jsonl"
        s3.upload_file(output_path, bucket, key)
        print(f"[shard {shard_index}] uploaded to s3://{bucket}/{key}")


if __name__ == "__main__":
    main()
