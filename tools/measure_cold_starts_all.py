import json
import statistics
import subprocess
import time
from datetime import datetime

NAMESPACE = "app"
SELECTOR = "app=parser-service"
TRIALS = 3
OUTPUT = "results/cold_start_three_trials.json"


def kubectl(*args):
    return subprocess.check_output(
        ["kubectl", *args], text=True
    )


def get_pods():
    data = json.loads(kubectl(
        "-n", NAMESPACE, "get", "pods",
        "-l", SELECTOR, "-o", "json"
    ))
    return data["items"]


def parse_time(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def ready_time(pod):
    for condition in pod.get("status", {}).get("conditions", []):
        if condition["type"] == "Ready" and condition["status"] == "True":
            return parse_time(condition["lastTransitionTime"])
    return None


results = []

for trial in range(1, TRIALS + 1):
    old_names = {p["metadata"]["name"] for p in get_pods()}

    print(f"\n========== COLD START TRIAL {trial} ==========")
    print("Deleting every application Pod...")

    subprocess.run([
        "kubectl", "-n", NAMESPACE, "delete", "pods",
        "-l", SELECTOR, "--wait=true", "--timeout=180s"
    ], check=True)

    deadline = time.monotonic() + 300
    measurements = []

    while time.monotonic() < deadline:
        pods = [
            p for p in get_pods()
            if p["metadata"]["name"] not in old_names
        ]

        ready_pods = [
            p for p in pods if ready_time(p) is not None
        ]

        if len(ready_pods) >= 2:
            for pod in ready_pods:
                created = parse_time(pod["metadata"]["creationTimestamp"])
                ready = ready_time(pod)
                elapsed = (ready - created).total_seconds()

                measurements.append({
                    "pod": pod["metadata"]["name"],
                    "creation_to_ready_seconds": round(elapsed, 3),
                })

            break

        time.sleep(2)

    if len(measurements) < 2:
        raise RuntimeError(f"Trial {trial}: two new Ready Pods not observed")

    print(json.dumps(measurements, indent=2))

    for item in measurements:
        name = item["pod"]
        logs = kubectl(
            "-n", NAMESPACE, "logs", name, "--tail=100"
        )
        item["download_logged"] = "Downloading model from" in logs
        item["loaded_logged"] = "Model loaded from object storage and ready" in logs
        print(
            name,
            "download_logged=", item["download_logged"],
            "loaded_logged=", item["loaded_logged"]
        )

    results.append({
        "trial": trial,
        "pods": measurements,
        "trial_median_seconds": round(
            statistics.median(
                x["creation_to_ready_seconds"] for x in measurements
            ), 3
        )
    })

all_times = [
    pod["creation_to_ready_seconds"]
    for trial in results
    for pod in trial["pods"]
]

summary = {
    "trials": results,
    "overall_pod_median_seconds": round(statistics.median(all_times), 3),
    "overall_pod_mean_seconds": round(statistics.mean(all_times), 3),
    "pod_measurements": len(all_times),
}

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n========== COLD START SUMMARY ==========")
print(json.dumps(summary, indent=2))
print("Saved:", OUTPUT)
