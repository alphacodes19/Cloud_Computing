#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${OLLAMA_HOST:-}" ]]; then
    if [[ -z "${OLLAMA_IP:-}" ]]; then
        DEFAULT_IFACE="$(
            ip -4 route show default |
            awk 'NR==1 {print $5}'
        )"

        if [[ -z "$DEFAULT_IFACE" ]]; then
            echo "ERROR: Could not detect the default network interface." >&2
            exit 1
        fi

        OLLAMA_IP="$(
            ip -o -4 addr show dev "$DEFAULT_IFACE" scope global |
            awk 'NR==1 {split($4, a, "/"); print a[1]}'
        )"
    fi

    if [[ -z "$OLLAMA_IP" || "$OLLAMA_IP" == 127.* ]]; then
        echo "ERROR: Could not detect a valid WSL IPv4 address." >&2
        exit 1
    fi

    OLLAMA_HOST="http://${OLLAMA_IP}:11434"
fi

export OLLAMA_HOST

if [[ ! "$OLLAMA_HOST" =~ ^https?://[^[:space:]]+$ ]]; then
    echo "ERROR: Invalid OLLAMA_HOST: $OLLAMA_HOST" >&2
    exit 1
fi

echo "Using Ollama endpoint: ${OLLAMA_HOST}" >&2

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    KUBECTL_ARGS=(apply --dry-run=client -f -)
else
    KUBECTL_ARGS=(apply -f -)
fi

python - <<'PY' | kubectl "${KUBECTL_ARGS[@]}"
import os
import sys
from pathlib import Path

manifest = Path("k8s/label-jobs.yaml").read_text()
placeholder = "${OLLAMA_HOST}"

if manifest.count(placeholder) != 2:
    raise SystemExit("Expected exactly two OLLAMA_HOST placeholders.")

sys.stdout.write(manifest.replace(placeholder, os.environ["OLLAMA_HOST"]))
PY
