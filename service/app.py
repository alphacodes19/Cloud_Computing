"""Task 3: inference service with controlled CPU inference concurrency."""
import json
import os
import socket
import threading

import boto3
import torch
from fastapi import FastAPI, Response
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_URI = os.environ["MODEL_URI"]
MODEL_VERSION = os.environ["MODEL_VERSION"]
S3_ENDPOINT = os.environ["S3_ENDPOINT"]
MAX_NEW_TOKENS = int(os.environ["MAX_NEW_TOKENS"])
FAULT_INJECT = os.environ.get("FAULT_INJECT", "0") == "1"

SCHEMA_PATH = os.environ.get("SCHEMA_PATH", "data/schema.json")
LOCAL_MODEL_DIR = "/tmp/model"

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

app = FastAPI()
_state = {"ready": False, "model": None, "tokenizer": None, "schema": None}
_inference_lock = threading.Lock()


def download_model():
    """Download model and tokenizer files from S3-compatible storage."""
    bucket, _, prefix = MODEL_URI.replace("s3://", "").partition("/")
    s3 = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("S3_SECRET_KEY"),
    )

    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):].lstrip("/")
            if not rel:
                continue

            dest = os.path.join(LOCAL_MODEL_DIR, rel)
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            s3.download_file(bucket, key, dest)


def load_model_in_background():
    try:
        print(f"Downloading model from {MODEL_URI}", flush=True)
        download_model()

        with open(SCHEMA_PATH, encoding="utf-8") as f:
            _state["schema"] = json.load(f)

        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_DIR)
        model.eval()

        _state["tokenizer"] = tokenizer
        _state["model"] = model
        _state["ready"] = True

        print("Model loaded from object storage and ready", flush=True)

    except Exception as exc:  # noqa: BLE001
        print(f"Model load FAILED: {exc}", flush=True)


@app.on_event("startup")
def startup():
    threading.Thread(target=load_model_in_background, daemon=True).start()


class ParseRequest(BaseModel):
    text: str


def validate_output(obj, schema):
    if not isinstance(obj, dict):
        return False

    if set(obj.keys()) != set(schema["keys"]):
        return False

    if obj["intent"] not in schema["intents"]:
        return False

    for key in ("date", "time", "place"):
        if obj[key] is not None and not isinstance(obj[key], str):
            return False

    return True


def run_model(text):
    tokenizer = _state["tokenizer"]
    model = _state["model"]

    prompt = f"Command: {text}\nJSON:"
    inputs = tokenizer(prompt, return_tensors="pt")

    input_length = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_ids = output_ids[0, input_length:]
    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()


@app.get("/healthz")
def healthz():
    return Response(status_code=200)


@app.get("/readyz")
def readyz():
    return Response(status_code=200 if _state["ready"] else 503)


@app.post("/parse")
def parse(req: ParseRequest, response: Response):
    response.headers["X-Model-Version"] = MODEL_VERSION
    response.headers["X-Served-By"] = socket.gethostname()

    if not _state["ready"]:
        response.status_code = 503
        return {"error": "model_not_ready"}

    if FAULT_INJECT:
        raw_output = "not json"
    else:
        with _inference_lock:
            raw_output = run_model(req.text)

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError as exc:
        print(
            f"Invalid JSON from model version {MODEL_VERSION}: "
            f"{exc.msg}, position={exc.pos}",
            flush=True,
        )
        parsed = None

    if parsed is None or not validate_output(parsed, _state["schema"]):
        print(
            f"Model output failed schema validation: version={MODEL_VERSION}",
            flush=True,
        )
        response.status_code = 500
        return {"error": "invalid_model_output"}

    return parsed
