"""Generate predictions.jsonl for score.py, from either:
  --model-dir PATH        a local HF model (trained student, or untrained pythia-70m)
  --teacher                the Ollama teacher (phi3.5), same prompt/settings as Task 1

Usage:
  python training/predict.py --teacher --out results/teacher_predictions.jsonl
  python training/predict.py --model-dir EleutherAI/pythia-70m --out results/untrained_predictions.jsonl
  python training/predict.py --model-dir models/v1 --out results/student_v1_predictions.jsonl

Then score with:
  python tools/score.py results/teacher_predictions.jsonl data/validation.jsonl \
      --out results/teacher_validation.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def predict_with_teacher(commands):
    from worker.label import call_teacher, load_prompt_template  # noqa
    prompt_template = load_prompt_template("prompts/teacher_prompt.txt")
    for cmd_id, text in commands:
        yield cmd_id, call_teacher(prompt_template, text)


def predict_with_hf_model(model_dir, commands, max_new_tokens):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.eval()

    for cmd_id, text in commands:
        prompt = f"Command: {text}\nJSON:"
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        full_text = tokenizer.decode(out[0], skip_special_tokens=True)
        generated = full_text[len(prompt):].strip()
        yield cmd_id, generated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation", default="data/validation.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-dir")
    ap.add_argument("--teacher", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    if not args.teacher and not args.model_dir:
        sys.exit("pass either --teacher or --model-dir")

    commands = []
    with open(args.validation, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            commands.append((rec["id"], rec["text"]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as out:
        gen = predict_with_teacher(commands) if args.teacher else \
            predict_with_hf_model(args.model_dir, commands, args.max_new_tokens)
        for cmd_id, prediction in gen:
            out.write(json.dumps({"id": cmd_id, "prediction": prediction}) + "\n")
            print(f"{cmd_id}: {prediction[:80]}")

    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
