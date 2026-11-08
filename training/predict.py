"""Generate validation predictions from Phi-3.5 or a Hugging Face causal LM.

Examples:
  python training/predict.py --teacher --out results/teacher_predictions.jsonl
  python training/predict.py --model-dir EleutherAI/pythia-70m --out results/untrained_predictions.jsonl
  python training/predict.py --model-dir models/v1 --out results/student_v1_predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_validation(path: str) -> list[tuple[str, str]]:
    commands: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record.get("id"), str) or not isinstance(record.get("text"), str):
                raise ValueError(f"Invalid validation record at line {line_number}")
            commands.append((record["id"], record["text"]))
    return commands


def predict_with_teacher(commands: list[tuple[str, str]]):
    from worker.label import call_teacher, load_json_schema, load_prompt_template

    prompt_template = load_prompt_template(str(ROOT / "prompts/teacher_prompt.txt"))
    json_schema = load_json_schema(str(ROOT / "data/schema.json"))
    for command_id, text in commands:
        yield command_id, call_teacher(prompt_template, text, json_schema)


def predict_with_hf_model(model_dir: str, commands: list[tuple[str, str]], max_new_tokens: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer does not define an EOS token")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model.to(torch.device("cpu"))
    model.eval()

    for command_id, text in commands:
        prompt = f"Command: {text}\nJSON:"
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        continuation = generated[0, inputs["input_ids"].shape[1]:]
        prediction = tokenizer.decode(continuation, skip_special_tokens=True).strip()
        yield command_id, prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation", default="data/validation.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-dir")
    parser.add_argument("--teacher", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()

    if args.teacher == bool(args.model_dir):
        parser.error("Pass exactly one of --teacher or --model-dir")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")

    commands = load_validation(str(ROOT / args.validation) if not Path(args.validation).is_absolute() else args.validation)
    output_path = Path(args.out)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.teacher:
        predictions = predict_with_teacher(commands)
    else:
        model_dir = args.model_dir
        if not Path(model_dir).is_absolute() and (ROOT / model_dir).exists():
            model_dir = str(ROOT / model_dir)
        predictions = predict_with_hf_model(model_dir, commands, args.max_new_tokens)

    with output_path.open("w", encoding="utf-8") as output:
        for command_id, prediction in predictions:
            output.write(
                json.dumps(
                    {"id": command_id, "prediction": prediction},
                    ensure_ascii=False,
                ) + "\n"
            )
            print(f"{command_id}: {prediction[:100]}", flush=True)

    print(f"Wrote {len(commands)} predictions to {output_path}", flush=True)


if __name__ == "__main__":
    main()
