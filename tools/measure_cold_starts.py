import csv
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


OUTPUT = Path("results/kubernetes/cold-start-trials.csv")
SUMMARY = Path("results/kubernetes/cold-start-summary.json")


def kubectl(*args):
    result = subprocess.run(
        ["kubectl", "-n", "app", *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def get_pods():
    return json.loads(
        kubectl("get", "pods", "-l", "app=parser-service", "-o", "json")
    )["items"]


def is_ready(pod):
    conditions = pod.get("status", {}).get("conditions", [])
    return (
        pod.get("status", {}).get("phase") == "Running"
        and any(
            condition.get("type") == "Ready"
            and condition.get("status") == "True"
            for condition in conditions
        )
    )


def ready_pods():
    return [pod for pod in get_pods() if is_ready(pod)]


OUTPUT.parent.mkdir(parents=True, exist_ok=True)
records = []

for trial in range(1, 4):
    current = ready_pods()

    if len(current) != 2:
        raise SystemExit(
            f"Trial {trial}: expected 2 Ready Pods, found {len(current)}."
        )

    victim = sorted(current, key=lambda p: p["metadata"]["name"])[0]
    old_names = {p["metadata"]["name"] for p in get_pods()}
    victim_name = victim["metadata"]["name"]

    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()

    print(f"\nTrial {trial}/3: deleting {victim_name}")
    kubectl("delete", "pod", victim_name, "--wait=false")

    replacement = None
    deadline = time.perf_counter() + 360

    while time.perf_counter() < deadline:
        candidates = [
            pod for pod in get_pods()
            if pod["metadata"]["name"] not in old_names
        ]

        ready_candidates = [pod for pod in candidates if is_ready(pod)]

        if ready_candidates:
            replacement = max(
                ready_candidates,
                key=lambda p: p["metadata"].get("creationTimestamp", ""),
            )
            break

        time.sleep(2)

    if replacement is None:
        raise SystemExit(
            f"Trial {trial}: replacement did not become Ready within 360 seconds."
        )

    elapsed = time.perf_counter() - started
    ready_at = datetime.now(timezone.utc).isoformat()

    record = {
        "trial": trial,
        "deleted_pod": victim_name,
        "replacement_pod": replacement["metadata"]["name"],
        "started_at_utc": started_at,
        "ready_at_utc": ready_at,
        "cold_start_seconds": round(elapsed, 3),
    }
    records.append(record)

    print(
        f"Replacement {record['replacement_pod']} Ready "
        f"in {record['cold_start_seconds']} seconds."
    )

    remaining = ready_pods()
    if len(remaining) != 2:
        raise SystemExit(
            f"After trial {trial}, expected 2 Ready Pods; found {len(remaining)}."
        )


with OUTPUT.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)

times = [row["cold_start_seconds"] for row in records]

summary = {
    "trials": len(times),
    "measure_definition": (
        "Time from Pod deletion request until replacement Pod Ready condition"
    ),
    "median_cold_start_seconds": round(statistics.median(times), 3),
    "mean_cold_start_seconds": round(statistics.mean(times), 3),
    "minimum_cold_start_seconds": round(min(times), 3),
    "maximum_cold_start_seconds": round(max(times), 3),
    "measurements": records,
}

SUMMARY.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

print("\n========== COLD-START SUMMARY ==========")
print(json.dumps(summary, indent=2))
print(f"\nSaved: {OUTPUT}")
print(f"Saved: {SUMMARY}")
