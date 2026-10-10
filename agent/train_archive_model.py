"""Optional Unsloth vision imitation training on the verified archive export."""
import argparse
import json
from pathlib import Path

try:
    from .archive_data import conversation, controls, lines, safe_frame
except ImportError:
    from archive_data import conversation, controls, lines, safe_frame
from PIL import Image
import hashlib


def check_archive(folder):
    partitions = {name: list(lines(folder / f"{name}.jsonl")) for name in ("train", "validation")}
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    for name, rows in partitions.items():
        if not rows or len(rows) != report[name]["rows"]:
            raise ValueError("empty partition or report count mismatch")
        if sorted({r["recording"] for r in rows}) != report[name]["recordings"]:
            raise ValueError("recording manifest mismatch")
        if sorted({r["group"] for r in rows}) != report[name]["groups"]:
            raise ValueError("holdout group mismatch")
        for row in rows:
            controls(row["controls"])
            if row["split"] != name:
                raise ValueError("partition label mismatch")
            with Image.open(safe_frame(folder, row["image"])) as image:
                if hashlib.sha256(image.convert("RGB").tobytes()).hexdigest() != row["image_sha256"]:
                    raise ValueError("image content hash mismatch")
    train, validation = partitions["train"], partitions["validation"]
    if ({r["group"] for r in train} & {r["group"] for r in validation}
            or {r["image_sha256"] for r in train} & {r["image_sha256"] for r in validation}):
        raise ValueError("train/validation overlap")
    return partitions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--model", help="explicit local or Hugging Face vision model")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("output/archive-model"))
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()
    partitions = check_archive(args.data)
    if args.check_data:
        print(json.dumps({"valid": True, **{name: {"rows": len(rows), "fire_positive": sum(r["controls"]["fire"] for r in rows)}
                                               for name, rows in partitions.items()}}))
        return
    if not args.model or args.steps <= 0:
        parser.error("choose --model and positive --steps")
    for rows in partitions.values():
        if {r["controls"]["fire"] for r in rows} != {0, 1}:
            parser.error("both partitions must contain firing and non-firing examples")
    from unsloth import FastVisionModel
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    model, processor = FastVisionModel.from_pretrained(
        args.model, load_in_4bit=False, local_files_only=args.local_files_only,
        use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model, finetune_vision_layers=True, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=8, lora_alpha=8, lora_dropout=0, bias="none", random_state=7,
    )
    FastVisionModel.for_training(model)
    datasets = {name: [conversation(row, args.data) for row in rows] for name, rows in partitions.items()}
    trainer = SFTTrainer(
        model=model, processing_class=processor,
        data_collator=UnslothVisionDataCollator(model, processor),
        train_dataset=datasets["train"], eval_dataset=datasets["validation"],
        args=SFTConfig(
            output_dir=str(args.output / "checkpoints"), max_steps=args.steps,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=4, learning_rate=2e-4,
            bf16=False, fp16=False, report_to="none", seed=7,
            remove_unused_columns=False, dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True}, max_length=2048,
        ),
    )
    before = trainer.evaluate()
    trainer.train()
    after = trainer.evaluate()
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output / "adapter"))
    processor.save_pretrained(str(args.output / "adapter"))
    (args.output / "evaluation.json").write_text(json.dumps({"before": before, "after": after,
        "claim": "held-out imitation loss only; no gameplay promotion", "steps": args.steps}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
