"""Task 3: inference service.

Config comes only from four environment variables:
  MODEL_URI        s3://bucket/key-prefix pointing at the model+tokenizer
  MODEL_VERSION    an arbitrary version string, echoed in X-Model-Version
  S3_ENDPOINT      SeaweedFS (S3-compatible) endpoint URL
  MAX_NEW_TOKENS   generation length cap

Credentials (S3_ACCESS_KEY / S3_SECRET_KEY) come from a mounted Kubernetes
Secret, read here only as environment variables set by that Secret — never
hardcoded and never committed.

FAULT_INJECT=1 makes /parse replace the model's real output with the literal
text "not json", before validation, to test the 500 path end to end.
"""
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

app = FastAPI()
_state = {"ready": False, "model": None, "tokenizer": None, "schema": None}


def download_model():
    """Pull the model+tokenizer files from object storage into LOCAL_MODEL_DIR."""
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
        print("model loaded and ready")
    except Exception as exc:
        print(f"model load FAILED: {exc}")


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
    tokenizer, model = _state["tokenizer"], _state["model"]
    prompt = f"Command: {text}\nJSON:"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    full_text = tokenizer.decode(out[0], skip_special_tokens=True)
    return full_text[len(prompt):].strip()


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

    raw_output = "not json" if FAULT_INJECT else run_model(req.text)

    try:
        parsed = json.loads(raw_output)
    except Exception:
        parsed = None

    if parsed is None or not validate_output(parsed, _state["schema"]):
        response.status_code = 500
        return {"error": "invalid_model_output"}

    return parsed
