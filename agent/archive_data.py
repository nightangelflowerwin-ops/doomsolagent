"""Prepare verified DoomSol human recordings for optional Unsloth vision SFT."""
import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image

HOSTS = {"doomsol.com", "www.doomsol.com", "doomsol.io", "www.doomsol.io"}
AXES = ("move_x", "move_y", "look_x", "look_y")
BUTTONS = ("fire", "use", "map", "escape")
PROMPT = ("Imitate the recorded DoomSol keyboard and mouse controls for this gameplay image. "
          "Return JSON with move_x (A=-1,D=1), move_y (W=-1,S=1), "
          "look_x (mouse dx/180), look_y (mouse dy/120), all clipped to [-1,1], "
          "and fire, use (Space), map (Tab), escape as 0 or 1. "
          "These are demonstrations, not guaranteed optimal actions.")


def lines(path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def number(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError("invalid numeric control")
    return value


def controls(row):
    result = {name: round(number(row[name]), 6) for name in AXES}
    if any(abs(value) > 1 for value in result.values()):
        raise ValueError("axis outside [-1,1]")
    for name in BUTTONS:
        value = row[name]
        if value not in (0, 1):
            raise ValueError("invalid binary control")
        result[name] = int(value)
    return result


def input_controls(sample):
    """Keep original batch semantics; do not invent target aim from mouse movement."""
    if not sample.get("pointerLocked") or "heldKeys" not in sample:
        raise ValueError("no locked keyboard recording")
    events = sample.get("events", [])
    # Long accumulated event batches cannot be aligned to one image reliably.
    times = [number(e["t"]) for e in events if "t" in e]
    if times and max(times) - min(times) > 850:
        raise ValueError("long accumulated event batch")
    held = set(sample["heldKeys"])
    down = {e.get("code") for e in events if e.get("kind") == "key" and e.get("down")}
    pointer = [e for e in events if e.get("kind") == "pointer"]
    return controls({
        "move_x": int("KeyD" in held) - int("KeyA" in held),
        "move_y": int("KeyS" in held) - int("KeyW" in held),
        "look_x": max(-1, min(1, sum(number(e.get("dx", 0)) for e in pointer) / 180)),
        "look_y": max(-1, min(1, sum(number(e.get("dy", 0)) for e in pointer) / 120)),
        "fire": int(any(e.get("kind") == "button" and e.get("button") == 0 and e.get("down") for e in events)),
        "use": int("Space" in down), "map": int("Tab" in down), "escape": int("Escape" in down),
    })


def safe_frame(run, relative):
    path = (run / str(relative).replace("\\", "/")).resolve()
    if not path.is_relative_to(run.resolve()):
        raise ValueError("frame escapes replay")
    return path


def build_archive(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise ValueError("choose a fresh output directory to preserve previous exports")
    inventory, replay_info = [], {}
    for path in sorted((root / "replays").glob("*/meta.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        ids = set(meta.get("game_ids", []))
        accepted = (urlparse(meta.get("source_url") or "").hostname in HOSTS
                    and ids <= {"doomsol"} and meta.get("mode") == "human_demonstration")
        replay_info[path.parent.name] = (path.parent, meta, accepted)
    candidates, reserved, stats = [], set(), Counter()
    # Preserve every historical test frame, including copied model versions.
    for path in sorted((root / "models").rglob("*.jsonl")):
        teacher = path.parent.name == "dwasm_teacher"
        count = 0
        report_path = path.parent / "training_report.json"
        model_report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        reason = "simulator provenance and action/frame timing unverified" if teacher else ""
        if teacher:
            with path.open("rb") as handle:
                count = sum(1 for line in handle if line.strip())
            inventory.append({"file": path.relative_to(root).as_posix(), "rows": count, "exclusion": reason})
            continue
        for row in lines(path):
            count += 1
            run_name = row.get("run") or model_report.get("source_run") or model_report.get("source_replay")
            info = replay_info.get(run_name)
            if not info or not info[2]:
                stats["unverified_or_other_game_rows"] += 1
                reason = "missing DoomSol provenance or other game"
                continue
            try:
                frame = safe_frame(info[0], row["frame"])
                if path.name == "test_sequences.jsonl":
                    reserved.add(str(frame))
                action = controls(row)
                if not row.get("pointer_locked"):
                    raise ValueError("pointer unlocked")
                candidates.append((run_name, frame, number(row["wall_time"]), action, "derived_controls"))
            except (ValueError, KeyError, TypeError):
                stats["unsupported_derived_rows"] += 1
        inventory.append({"file": path.relative_to(root).as_posix(), "rows": count, "exclusion": reason or None})
    for run_name, (run, meta, accepted) in sorted(replay_info.items()):
        path = run / "inputs.jsonl"
        if not path.exists():
            continue
        frames = sorted(meta.get("frames", []), key=lambda f: f["timestamp"])
        timestamps = [number(f["timestamp"]) for f in frames]
        count = 0
        for sample in lines(path):
            count += 1
            if not accepted:
                continue
            try:
                t = number(sample["wallTime"])
                index = bisect_right(timestamps, t) - 1
                if index < 0 or t - timestamps[index] > 850:
                    raise ValueError("no recent preceding frame")
                frame = safe_frame(run, frames[index]["path"])
                candidates.append((run_name, frame, t, input_controls(sample), "raw_input_batch"))
            except (ValueError, KeyError, TypeError):
                stats["unaligned_or_unsupported_input_batches"] += 1
        inventory.append({"file": path.relative_to(root).as_posix(), "rows": count,
                          "exclusion": None if accepted else "missing DoomSol provenance or other game"})
    # Preserve historical test blocks and guard boundaries against alignment lag.
    bases = {run: min(r[2] for r in candidates if r[0] == run) for run in {r[0] for r in candidates}}
    reserved_blocks = {(run, int((t - bases[run]) // 20000))
                       for run, frame, t, _, _ in candidates if str(frame) in reserved}
    runs = {row[0] for row in candidates}
    validation_runs = set()
    if not reserved_blocks and len(runs) >= 2:
        validation_runs = {min(runs, key=lambda r: hashlib.sha256(r.encode()).hexdigest())}
    if not reserved_blocks and not validation_runs:
        raise ValueError("need historical test blocks or independent recording groups")
    output.mkdir(parents=True)
    (output / "images").mkdir()
    rows, seen, image_cache = [], set(), {}
    for run, frame, t, action, source in candidates:
        block = int((t - bases[run]) // 20000)
        split = "validation" if run in validation_runs or (run, block) in reserved_blocks else "train"
        if split == "train" and any((run, int((t + delta - bases[run]) // 20000)) in reserved_blocks
                                    for delta in (-850, 850)):
            stats["holdout_boundary_guard"] += 1
            continue
        try:
            if frame not in image_cache:
                with Image.open(frame) as image:
                    image = image.convert("RGB"); w, h = image.size
                    # Same legacy gameplay crop, excluding site controls and status panels.
                    image = image.crop((round(w*.165), round(h*.105), round(w*.835), round(h*.665)))
                    image = image.resize((384, 224), Image.Resampling.BILINEAR)
                    digest = hashlib.sha256(image.tobytes()).hexdigest()
                    image.save(output / "images" / f"{digest}.png")
                    image_cache[frame] = digest
            digest = image_cache[frame]
        except (OSError, ValueError):
            stats["missing_or_invalid_images"] += 1
            continue
        key = (split, digest, json.dumps(action, sort_keys=True))
        if key in seen:
            stats["duplicate_examples"] += 1
            continue
        seen.add(key)
        rows.append({"image": f"images/{digest}.png", "image_sha256": digest,
                     "controls": action, "recording": run, "timestamp": t,
                     "group": f"{run}:{block}" if reserved_blocks else run,
                     "label_source": source, "split": split})
    val_images = {r["image_sha256"] for r in rows if r["split"] == "validation"}
    kept = [r for r in rows if r["split"] == "validation" or r["image_sha256"] not in val_images]
    stats["train_images_overlapping_validation"] = len(rows) - len(kept)
    report = {"schema_version": 1, "sources": inventory, "counts": dict(stats),
              "split": "historical_20_second_blocks_with_850ms_guard" if reserved_blocks else "whole_recording",
              "limitations": ["Synchronized imitation, not proof of pre-action causal timing.",
                              "Derived labels retain legacy temporal dilation and alignment noise.",
                              "Single frames omit motion history; recordings may share game episodes.",
                              "Simulator samples and unverified game sources are inventoried but excluded."]}
    for split in ("train", "validation"):
        subset = [r for r in kept if r["split"] == split]
        if not subset:
            raise ValueError(f"empty {split} after image filtering")
        (output / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in subset), encoding="utf-8")
        report[split] = {"rows": len(subset), "fire_positive": sum(r["controls"]["fire"] for r in subset),
                         "recordings": sorted({r["recording"] for r in subset}), "groups": sorted({r["group"] for r in subset})}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def conversation(row, folder):
    """Only image and static control instructions are inputs; metadata stays out."""
    action = controls(row["controls"])
    path = safe_frame(Path(folder), row["image"])
    with Image.open(path) as image:
        image = image.convert("RGB").copy()
    return {"messages": [
        {"role": "user", "content": [{"type": "text", "text": PROMPT}, {"type": "image", "image": image}]},
        {"role": "assistant", "content": [{"type": "text", "text": json.dumps(action, sort_keys=True)}]},
    ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="local archive containing models/ and replays/")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_archive(args.archive, args.output)
    print(json.dumps({name: report[name] for name in ("train", "validation", "counts")}, indent=2))


if __name__ == "__main__":
    main()
