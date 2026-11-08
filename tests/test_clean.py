"""Unit tests for worker/clean.py's pure-logic helpers."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from worker.clean import norm, slot_appears_in_command


def test_norm_lowercases_and_collapses_whitespace():
    assert norm("  Seven   AM  ") == "seven am"


def test_slot_null_is_always_accepted():
    assert slot_appears_in_command(None, "anything at all") is True


def test_slot_value_must_appear_in_command():
    assert slot_appears_in_command("tomorrow", "set an alarm for tomorrow") is True
    assert slot_appears_in_command("friday", "set an alarm for tomorrow") is False


def test_slot_match_is_case_and_space_insensitive():
    assert slot_appears_in_command(
        "Seven AM", "set an alarm for seven   am please"
    )
