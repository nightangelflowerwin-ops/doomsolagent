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
    parser.add_argument("--epochs", type=float, default=1, help="complete passes over the training partition")
    parser.add_argument("--steps", type=int, help="explicit debug-only update cap; overrides epochs")
    parser.add_argument("--device", choices=("auto", "cpu"), default="auto")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    partitions = check_archive(args.data)
    if args.check_data:
        print(json.dumps({"valid": True, **{name: {"rows": len(rows), "fire_positive": sum(r["controls"]["fire"] for r in rows)}
                                               for name, rows in partitions.items()}}))
        return
    if not args.model or args.epochs <= 0 or (args.steps is not None and args.steps <= 0):
        parser.error("choose --model and positive --epochs or --steps")
    for rows in partitions.values():
        if {r["controls"]["fire"] for r in rows} != {0, 1}:
            parser.error("both partitions must contain firing and non-firing examples")
    import torch
    use_cpu = args.device == "cpu" or not torch.cuda.is_available()
    if use_cpu:
        train_cpu(args, partitions)
        return
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
            output_dir=str(args.output / "checkpoints"), max_steps=args.steps or -1,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=4, learning_rate=2e-4,
            bf16=False, fp16=False, report_to="none", seed=7,
            remove_unused_columns=False, dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True}, max_length=2048,
        ),
    )
    before = trainer.evaluate()
    trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    after = trainer.evaluate()
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output / "adapter"))
    processor.save_pretrained(str(args.output / "adapter"))
    (args.output / "evaluation.json").write_text(json.dumps({"before": before, "after": after,
        "claim": "held-out imitation loss only; no gameplay promotion", "epochs": args.epochs,
        "debug_step_cap": args.steps}, indent=2), encoding="utf-8")


def train_cpu(args, partitions):
    """Use the native reference backend when the vision GPU loader cannot run."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(7)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.float32, device_map="cpu",
        local_files_only=args.local_files_only, attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = get_peft_model(model, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0, bias="none", target_modules="all-linear",
    ))
    model.config.use_cache = False

    class RecordedImages(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return conversation(self.rows[index], args.data)

    def collate(rows):
        if len(rows) != 1:
            raise ValueError("CPU image collator requires batch size one")
        messages = rows[0]["messages"]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt", enable_thinking=False,
        )
        prefix = processor.apply_chat_template(
            messages[:-1], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", enable_thinking=False,
        )
        length = prefix["input_ids"].shape[1]
        if not torch.equal(inputs["input_ids"][:, :length], prefix["input_ids"]):
            raise ValueError("assistant prefix mismatch; refusing an incorrect loss mask")
        labels = inputs["input_ids"].clone()
        labels[:, :length] = -100
        if not (labels != -100).any():
            raise ValueError("empty assistant labels")
        inputs["labels"] = labels
        return inputs

    trainer = Trainer(
        model=model, processing_class=processor, data_collator=collate,
        train_dataset=RecordedImages(partitions["train"]),
        eval_dataset=RecordedImages(partitions["validation"]),
        args=TrainingArguments(
            output_dir=str(args.output / "checkpoints"), use_cpu=True,
            num_train_epochs=args.epochs, max_steps=args.steps or -1,
            per_device_train_batch_size=1, per_device_eval_batch_size=1,
            gradient_accumulation_steps=4, learning_rate=2e-4,
            lr_scheduler_type="cosine", warmup_ratio=.03,
            logging_steps=1, save_steps=100, save_total_limit=2,
            bf16=False, fp16=False, report_to="none", seed=7,
            remove_unused_columns=False, dataloader_pin_memory=False,
            optim="adamw_torch",
        ),
    )
    before = trainer.evaluate()
    trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
    after = trainer.evaluate()
    model.save_pretrained(str(args.output / "adapter"))
    processor.save_pretrained(str(args.output / "adapter"))
    (args.output / "evaluation.json").write_text(json.dumps({
        "before": before, "after": after, "epochs": args.epochs,
        "debug_step_cap": args.steps, "backend": "Transformers/PEFT CPU reference",
        "claim": "Full held-out imitation loss only; no gameplay promotion.",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
