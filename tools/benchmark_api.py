import argparse
import csv
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


def percentile(values, p):
    ordered = sorted(values)
    rank = max(1, math.ceil(p * len(ordered) / 100))
    return ordered[rank - 1]


def valid_frame(obj):
    if not isinstance(obj, dict):
        return False
    if set(obj) != {"intent", "date", "time", "place"}:
        return False
    if not isinstance(obj["intent"], str):
        return False
    return all(
        value is None or isinstance(value, str)
        for value in (obj["date"], obj["time"], obj["place"])
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    gold_path = Path("data/validation.jsonl")
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    gold = [
        json.loads(line)
        for line in gold_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if args.count < 1 or args.count > len(gold):
        raise SystemExit(f"Count must be between 1 and {len(gold)}.")

    samples = gold[:args.count]

    prediction_path = results_dir / "api_serial_50_predictions.jsonl"
    gold_subset_path = results_dir / "api_serial_50_gold.jsonl"
    csv_path = results_dir / "api_serial_50_latencies.csv"
    metrics_path = results_dir / "api_serial_50_metrics.json"

    latencies = []
    http_200 = 0
    http_errors = 0
    transport_errors = 0
    valid_json = 0
    valid_frames = 0

    started_all = time.perf_counter()

    with prediction_path.open("w", encoding="utf-8") as predictions, \
         csv_path.open("w", newline="", encoding="utf-8") as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "id", "http_status", "latency_ms",
                "valid_json", "valid_frame", "error"
            ],
        )
        writer.writeheader()

        for index, row in enumerate(samples, start=1):
            payload = json.dumps({"text": row["text"]}).encode("utf-8")
            request = urllib.request.Request(
                args.base_url.rstrip("/") + "/parse",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            start = time.perf_counter()
            status = None
            error = ""
            body = ""

            try:
                with urllib.request.urlopen(
                    request, timeout=args.timeout
                ) as response:
                    status = response.status
                    body = response.read().decode("utf-8")

            except urllib.error.HTTPError as exc:
                status = exc.code
                body = exc.read().decode("utf-8", errors="replace")
                error = f"HTTP {exc.code}"

            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                transport_errors += 1

            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies.append(elapsed_ms)

            if status == 200:
                http_200 += 1
            elif status is not None:
                http_errors += 1

            try:
                parsed = json.loads(body)
                json_ok = isinstance(parsed, dict)
            except (json.JSONDecodeError, TypeError):
                parsed = None
                json_ok = False

            frame_ok = valid_frame(parsed)

            if json_ok:
                valid_json += 1
            if frame_ok:
                valid_frames += 1

            predictions.write(json.dumps({
                "id": row["id"],
                "prediction": (
                    json.dumps(parsed, ensure_ascii=False)
                    if parsed is not None else body
                ),
            }, ensure_ascii=False) + "\n")

            writer.writerow({
                "id": row["id"],
                "http_status": status if status is not None else "ERROR",
                "latency_ms": round(elapsed_ms, 3),
                "valid_json": json_ok,
                "valid_frame": frame_ok,
                "error": error,
            })

            print(
                f"[{index}/{args.count}] {row['id']} "
                f"HTTP {status if status is not None else 'ERROR'} "
                f"{elapsed_ms:.1f} ms"
            )

    total_seconds = time.perf_counter() - started_all

    metrics = {
        "requests": args.count,
        "http_200": http_200,
        "http_errors": http_errors,
        "transport_errors": transport_errors,
        "valid_json_objects": valid_json,
        "valid_frames": valid_frames,
        "json_object_rate_percent": round(100 * valid_json / args.count, 2),
        "valid_frame_rate_percent": round(100 * valid_frames / args.count, 2),
        "mean_latency_ms": round(statistics.mean(latencies), 2),
        "p50_latency_ms": round(percentile(latencies, 50), 2),
        "p95_latency_ms": round(percentile(latencies, 95), 2),
        "min_latency_ms": round(min(latencies), 2),
        "max_latency_ms": round(max(latencies), 2),
        "total_duration_seconds": round(total_seconds, 2),
        "throughput_requests_per_second": round(
            args.count / total_seconds, 3
        ),
    }

    gold_subset_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples),
        encoding="utf-8",
    )

    metrics_path.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n========== BENCHMARK SUMMARY ==========")
    print(json.dumps(metrics, indent=2))
    print("\nSaved:")
    print(prediction_path)
    print(gold_subset_path)
    print(csv_path)
    print(metrics_path)


if __name__ == "__main__":
    main()
