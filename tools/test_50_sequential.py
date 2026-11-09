import json
import os
import statistics
import time
import urllib.error
import urllib.request

URL = "http://localhost:8080/parse"
INPUT = "data/validation.jsonl"
OUTPUT = "results/task3-50-sequential-decode-fix.json"

with open(INPUT, encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()][:50]

if len(records) != 50:
    raise SystemExit(f"Expected 50 records, found {len(records)}")

times = []
failures = []
versions = {}

for i, record in enumerate(records, 1):
    payload = json.dumps({"text": record["text"]}).encode()
    request = urllib.request.Request(
        URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            status = response.status
            body = response.read().decode()
            version = response.headers.get("X-Model-Version", "unknown")
    except urllib.error.HTTPError as exc:
        status = exc.code
        body = exc.read().decode()
        version = exc.headers.get("X-Model-Version", "unknown")
    except Exception as exc:
        status = 0
        body = str(exc)
        version = "unknown"

    elapsed = time.monotonic() - start
    times.append(elapsed)
    versions[version] = versions.get(version, 0) + 1

    if status != 200:
        failures.append({
            "index": i,
            "status": status,
            "body": body[:300],
        })

    print(f"{i:02d}/50  HTTP {status}  {elapsed:.2f}s", flush=True)

ordered = sorted(times)
p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]

result = {
    "total": len(records),
    "successful": len(records) - len(failures),
    "failed": len(failures),
    "failure_rate_percent": round(100 * len(failures) / len(records), 2),
    "mean_seconds": round(statistics.mean(times), 3),
    "p50_seconds": round(statistics.median(times), 3),
    "p95_seconds": round(p95, 3),
    "max_seconds": round(max(times), 3),
    "model_versions": versions,
    "failures": failures,
}

os.makedirs("results", exist_ok=True)
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print("\n========== FINAL RESULTS ==========")
print(json.dumps(result, indent=2))
print(f"\nSaved: {OUTPUT}")
