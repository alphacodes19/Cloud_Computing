"""Task 1 step 6: merge shard outputs into datasets/v1/train.jsonl.

Usage:
  python worker/clean.py datasets/teacher/shard-*.jsonl \
      --out datasets/v1/train.jsonl --schema data/schema.json

Removes, in this order, and counts each reason separately:
  (a) replies that are not valid JSON
  (b) replies whose keys are not exactly intent, date, time, place
  (c) replies whose intent is not one of the eight
  (d) replies with a slot value that is neither null nor text appearing
      in the command (case-insensitive, whitespace-insensitive substring)
  duplicate commands (same "text"); keeps the one with the smallest id

Keeps surviving records sorted by id, writes the SHA-256 checksum of the
output file next to it, and prints the counts table required in the report.
"""
import argparse
import glob
import hashlib
import json
import re
import sys


REQUIRED_KEYS = {"intent", "date", "time", "place"}


def norm(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def slot_appears_in_command(value, command_text):
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    return norm(value) in norm(command_text)


def load_schema(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+", help="shard-*.jsonl files (glob patterns ok)")
    ap.add_argument("--out", default="datasets/v1/train.jsonl")
    ap.add_argument("--schema", default="data/schema.json")
    args = ap.parse_args()

    schema = load_schema(args.schema)
    valid_intents = set(schema["intents"])

    shard_files = []
    for pattern in args.shards:
        shard_files.extend(sorted(glob.glob(pattern)))
    if not shard_files:
        sys.exit("no shard files matched")

    counts = {"input": 0, "bad_json": 0, "bad_keys": 0, "bad_intent": 0,
              "bad_slot": 0, "duplicates": 0, "kept": 0}

    seen_commands = {}  # normalized command text -> smallest id kept so far
    kept_by_id = {}

    for path in shard_files:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                counts["input"] += 1
                rec = json.loads(line)  # shard lines themselves are trusted JSON
                cmd_id, text, raw = rec["id"], rec["text"], rec["teacher_raw"]

                try:
                    reply = json.loads(raw)
                except Exception:
                    counts["bad_json"] += 1
                    continue

                if not isinstance(reply, dict) or set(reply.keys()) != REQUIRED_KEYS:
                    counts["bad_keys"] += 1
                    continue

                if reply["intent"] not in valid_intents:
                    counts["bad_intent"] += 1
                    continue

                if not all(slot_appears_in_command(reply[k], text) for k in ("date", "time", "place")):
                    counts["bad_slot"] += 1
                    continue

                norm_text = norm(text)
                if norm_text in seen_commands:
                    counts["duplicates"] += 1
                    existing_id = seen_commands[norm_text]
                    if cmd_id < existing_id:
                        del kept_by_id[existing_id]
                        seen_commands[norm_text] = cmd_id
                        kept_by_id[cmd_id] = {"id": cmd_id, "text": text, **reply}
                    continue

                seen_commands[norm_text] = cmd_id
                kept_by_id[cmd_id] = {"id": cmd_id, "text": text, **reply}

    counts["kept"] = len(kept_by_id)

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for cmd_id in sorted(kept_by_id):
            f.write(json.dumps(kept_by_id[cmd_id]) + "\n")

    digest = hashlib.sha256()
    with open(args.out, "rb") as f:
        digest.update(f.read())
    sha_path = args.out.rsplit(".", 1)[0] + ".sha256"
    with open(sha_path, "w", encoding="utf-8") as f:
        f.write(digest.hexdigest() + "\n")

    check = (counts["bad_json"] + counts["bad_keys"] + counts["bad_intent"]
             + counts["bad_slot"] + counts["duplicates"] + counts["kept"])
    print(json.dumps(counts, indent=2))
    print(f"sha256: {digest.hexdigest()}")
    print(f"sanity check: input ({counts['input']}) == sum of reasons ({check}): "
          f"{counts['input'] == check}")


if __name__ == "__main__":
    main()
