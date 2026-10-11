"""Train an optional Unsloth action classifier; never send gameplay input."""
import argparse
import json
from pathlib import Path

from decision_data import QUESTIONS


def load_rows(path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"empty dataset: {path.name}")
    for row in rows:
        if (not isinstance(row.get("state"), str) or row.get("questions") != QUESTIONS
                or row.get("gold", {}).get("action") not in QUESTIONS["action"]["criteria"]):
            raise ValueError("dataset does not match the DoomSol action schema")
    return rows


def check_data(folder):
    train, validation = load_rows(folder / "train.jsonl"), load_rows(folder / "validation.jsonl")
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    groups = report["groups"]
    if (not groups["train"] or not groups["validation"]
            or set(groups["train"]) & set(groups["validation"])):
        raise ValueError("replay groups overlap or are empty")
    if {row["state"] for row in train} & {row["state"] for row in validation}:
        raise ValueError("training observations overlap validation")
    if len(train) != report["train_rows"] or len(validation) != report["validation_rows"]:
        raise ValueError("dataset counts do not match the export report")
    return train, validation


def missing_actions(rows):
    return sorted(set(QUESTIONS["action"]["criteria"]) - {row["gold"]["action"] for row in rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("output/decision-data"))
    parser.add_argument("--model", help="local model folder or explicitly chosen Hugging Face model")
    parser.add_argument("--output", type=Path, default=Path("output/decision-model"))
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--check-data", action="store_true", help="validate data without importing Unsloth or loading weights")
    args = parser.parse_args()
    train, validation = check_data(args.data)
    if args.check_data:
        print(json.dumps({"train_rows": len(train), "validation_rows": len(validation), "valid": True,
                          "missing_train_actions": missing_actions(train),
                          "missing_validation_actions": missing_actions(validation)}))
        return
    if not args.model or args.steps <= 0:
        parser.error("training requires --model and a positive --steps")
    if missing_actions(train) or missing_actions(validation):
        parser.error(f"record more independent demonstrations first: train missing {missing_actions(train)}, "
                     f"validation missing {missing_actions(validation)}")
    # Keep optional model dependencies out of passive capture and the CPU test suite.
    from unsloth import FastDecisionModel, DecisionTrainer
    from transformers import TrainingArguments

    model, tokenizer = FastDecisionModel.from_pretrained(
        model_name=args.model, max_seq_length=2048, load_in_4bit=False,
        local_files_only=args.local_files_only,
    )
    model = FastDecisionModel.get_peft_model(model, r=8, lora_alpha=8, random_state=7)
    train_items, train_report = FastDecisionModel.build_dataset(train, tokenizer, model)
    validation_items, validation_report = FastDecisionModel.build_dataset(validation, tokenizer, model)
    for report in (train_report, validation_report):
        if report["skipped"] or report["truncated"]:
            raise ValueError(f"Unsloth rejected or truncated data: {report}")
    before = FastDecisionModel.evaluate(model, tokenizer, validation_items)
    trainer = DecisionTrainer(
        model=model, processing_class=tokenizer, train_dataset=train_items,
        args=TrainingArguments(
            output_dir=str(args.output / "checkpoints"), max_steps=args.steps,
            per_device_train_batch_size=1, gradient_accumulation_steps=4,
            learning_rate=2e-4, logging_steps=10, report_to="none", seed=7,
            bf16=False, fp16=False,
        ),
    )
    trainer.train()
    after = FastDecisionModel.evaluate(model, tokenizer, validation_items)
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output / "model"))
    (args.output / "evaluation.json").write_text(json.dumps({
        "before": before, "after": after, "steps": args.steps,
        "claim": "held-out controller imitation only; no gameplay promotion",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
