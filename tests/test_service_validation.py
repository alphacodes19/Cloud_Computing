"""Unit tests for service/app.py's validate_output()."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "service"))

SCHEMA = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "schema.json")))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "app", os.path.join(os.path.dirname(__file__), "..", "service", "app.py")
)
# app.py reads required env vars at import time; set dummies for the test import.
os.environ.setdefault("MODEL_URI", "s3://models/v1")
os.environ.setdefault("MODEL_VERSION", "test")
os.environ.setdefault("S3_ENDPOINT", "http://localhost:8333")
os.environ.setdefault("MAX_NEW_TOKENS", "64")
app_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_module)


def test_valid_output_passes():
    obj = {"intent": "alarm_set", "date": "tomorrow", "time": "seven am", "place": None}
    assert app_module.validate_output(obj, SCHEMA) is True


def test_wrong_keys_fail():
    obj = {"intent": "alarm_set", "date": "tomorrow", "time": "seven am"}
    assert app_module.validate_output(obj, SCHEMA) is False


def test_bad_intent_fails():
    obj = {"intent": "not_real", "date": None, "time": None, "place": None}
    assert app_module.validate_output(obj, SCHEMA) is False
