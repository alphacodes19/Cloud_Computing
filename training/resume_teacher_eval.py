"""Resumable Phi-3.5 validation evaluation for CS3081."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_validation(path):
    records = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    ids = [r["id"] for r in records]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate validation IDs")

    return records


def load_existing(path, valid_ids):
    completed = {}

    if not path.exists():
        return completed

    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                command_id = record["id"]
                prediction = record["prediction"]

                if command_id not in valid_ids:
                    raise ValueError(f"Unknown ID: {command_id}")

                if not isinstance(prediction, str):
                    raise ValueError("Prediction must be a string")

                if command_id in completed:
                    raise ValueError(f"Duplicate ID: {command_id}")

                completed[command_id] = prediction

            except Exception as exc:
                raise ValueError(
                    f"Invalid checkpoint line {line_no}: {exc}"
                ) from exc

    return completed


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--validation",
        default="data/validation.jsonl"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--attempts", type=int, default=1)

    args = parser.parse_args()

    if args.timeout <= 0 or args.attempts < 1:
        parser.error("Timeout and attempts must be positive")

    # Set these BEFORE importing worker.label.
    os.environ["REQUEST_TIMEOUT"] = str(args.timeout)
    os.environ["MAX_ATTEMPTS"] = str(args.attempts)

    from worker.label import (
        call_teacher,
        load_prompt_template,
        load_json_schema,
    )

    validation_path = Path(args.validation)
    if not validation_path.is_absolute():
        validation_path = ROOT / validation_path

    output_path = Path(args.out)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_validation(validation_path)
    valid_ids = {r["id"] for r in records}

    completed = load_existing(output_path, valid_ids)

    print(
        f"Validation records: {len(records)}\n"
        f"Already completed: {len(completed)}\n"
        f"Remaining: {len(records) - len(completed)}",
        flush=True,
    )

    prompt = load_prompt_template(
        str(ROOT / "prompts/teacher_prompt.txt")
    )

    schema = load_json_schema(
        str(ROOT / "data/schema.json")
    )

    error_path = output_path.with_name(
        output_path.name + ".failures.jsonl"
    )

    failed_this_run = []

    # Append checkpoints after each successful prediction.
    with output_path.open("a", encoding="utf-8") as output:
        for index, record in enumerate(records, 1):
            command_id = record["id"]

            if command_id in completed:
                continue

            print(
                f"\n[{index}/{len(records)}] Generating {command_id}",
                flush=True,
            )

            try:
                raw = call_teacher(
                    prompt,
                    record["text"],
                    schema,
                )

                result = {
                    "id": command_id,
                    "prediction": raw,
                }

                output.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )
                output.flush()
                os.fsync(output.fileno())

                completed[command_id] = raw

                print(f"{command_id}: saved successfully", flush=True)

            except Exception as exc:
                failed_this_run.append(command_id)

                error_record = {
                    "id": command_id,
                    "error": str(exc),
                }

                with error_path.open("a", encoding="utf-8") as errors:
                    errors.write(
                        json.dumps(error_record) + "\n"
                    )

                print(
                    f"FAILED {command_id}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

                # Continue with the next validation command.
                continue

    missing = [
        record["id"]
        for record in records
        if record["id"] not in completed
    ]

    if missing:
        print(
            f"\nINCOMPLETE: {len(missing)} predictions remain.",
            flush=True,
        )
        print("Missing IDs:", ", ".join(missing), flush=True)
        print("Run this same command again to resume.", flush=True)
        return 2

    # Once complete, restore the original validation order.
    temporary = output_path.with_name(
        output_path.name + ".ordered.tmp"
    )

    with temporary.open("w", encoding="utf-8") as f:
        for record in records:
            item = {
                "id": record["id"],
                "prediction": completed[record["id"]],
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    temporary.replace(output_path)

    print(
        f"\nSUCCESS: {len(completed)} teacher predictions saved.",
        flush=True,
    )
    print(f"Output: {output_path}", flush=True)

    if failed_this_run:
        print(
            "Previously failed requests that succeeded on this run: "
            "they are now included in the output.",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
