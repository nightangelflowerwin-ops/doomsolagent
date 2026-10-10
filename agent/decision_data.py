"""Export closed-loop controller demonstrations for Unsloth decision training."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


QUESTIONS = {
    "action": {
        "type": "choice",
        "instructions": "Choose the next controller action from visual candidate telemetry.",
        "criteria": {
            "none": "Do not aim or fire.",
            "aim": "Adjust aim toward the tracked candidate.",
            "fire": "Tap fire at the tracked candidate.",
        },
    }
}


def observation_state(perception):
    """Allow only pre-decision features, without IDs, paths or outcome evidence."""
    resolution = perception.get("resolution")
    if (not isinstance(resolution, list) or len(resolution) != 2
            or any(not isinstance(v, (int, float)) or isinstance(v, bool)
                   or not math.isfinite(v) or v <= 0 for v in resolution)):
        raise ValueError("observation needs a positive resolution")
    width, height = resolution

    def number(value):
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value)):
            raise ValueError("non-finite or missing observation feature")
        return value

    target = perception.get("target")
    aim = perception.get("aim_error")
    features = {"resolution": resolution, "target": None, "aim_error": None}
    if target is not None:
        confidence = number(target.get("confidence"))
        age = number(target.get("age"))
        if not 0 <= confidence <= 1 or age < 1:
            raise ValueError("invalid candidate confidence or age")
        if not isinstance(aim, dict):
            raise ValueError("target has no aim evidence")
        features["target"] = {"confidence": confidence, "age": age}
        features["aim_error"] = {
            "dx": round(number(aim.get("dx")) / width, 6),
            "dy": round(number(aim.get("dy")) / height, 6),
            "distance": round(number(aim.get("distance")) / math.hypot(width, height), 6),
        }
    movement = perception.get("movement", {})
    features["movement"] = {
        "state": movement.get("state") if movement.get("state") in
        {"unknown", "stationary", "turning", "moving"} else "unknown",
        "global_change": number(movement.get("global_change")),
        "center_change": number(movement.get("center_change")),
    }
    return json.dumps(features, sort_keys=True, allow_nan=False)


def export_replay(meta):
    """Pair adjacent observation/decision events; labels imitate the controller."""
    if meta.get("schema_version") != 2 or meta.get("mode") != "closed_loop":
        raise ValueError("expected a schema 2 closed_loop replay")
    rows, skipped, pending = [], 0, None
    for event in meta.get("events", []):
        kind = event.get("kind")
        if kind == "observation":
            pending = event
        elif kind == "decision":
            action = event.get("action", {}).get("type")
            try:
                if pending is None or action not in QUESTIONS["action"]["criteria"]:
                    raise ValueError("unpaired or unsupported decision")
                state = observation_state(pending["perception"])
                rows.append({"state": state, "questions": QUESTIONS,
                             "gold": {"action": action}})
            except (KeyError, TypeError, AttributeError, ValueError):
                skipped += 1
            pending = None
        else:
            pending = None
    return rows, skipped


def write_dataset(paths, output, holdout_fraction=0.2, seed=7):
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout fraction must be between zero and one")
    groups, seen, skipped = [], set(), 0
    for path in paths:
        meta = json.loads(path.read_text(encoding="utf-8"))
        # Content identity makes copied replays one group regardless of filename.
        digest = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        rows, rejected = export_replay(meta)
        skipped += rejected
        if rows:
            groups.append((digest, rows))
    if len(groups) < 2:
        raise ValueError("need at least two distinct nonempty replays for a replay holdout")
    groups.sort(key=lambda group: hashlib.sha256(f"{seed}:{group[0]}".encode()).hexdigest())
    count = max(1, min(len(groups) - 1, round(len(groups) * holdout_fraction)))
    partitions = {"validation": groups[:count], "train": groups[count:]}
    # Repeated identical feature states across runs are also excluded from training.
    validation_states = {row["state"] for _, rows in partitions["validation"] for row in rows}
    train_rows = [row for _, rows in partitions["train"] for row in rows
                  if row["state"] not in validation_states]
    validation_rows = [row for _, rows in partitions["validation"] for row in rows]
    if not train_rows:
        raise ValueError("no training states remain after excluding validation duplicates")
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train_rows), ("validation", validation_rows)):
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    report = {
        "schema_version": 1, "label_source": "recorded_controller_demonstration",
        "split": "whole_replay_with_exact_state_overlap_removed", "seed": seed,
        "train_rows": len(train_rows), "validation_rows": len(validation_rows),
        "skipped_decisions": skipped,
        "removed_train_overlap": sum(len(rows) for _, rows in partitions["train"]) - len(train_rows),
        "action_counts": {
            name: dict(Counter(row["gold"]["action"] for row in rows))
            for name, rows in (("train", train_rows), ("validation", validation_rows))
        },
        "groups": {name: [digest for digest, _ in values] for name, values in partitions.items()},
        "limitations": ["Candidates are not confirmed enemies.",
                        "Controller actions are not human-reviewed optimal actions.",
                        "Validation measures imitation, not kills or level completion.",
                        "Separate replays may still come from the same game episode."],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replays", nargs="+", type=Path, help="closed-loop meta.json files")
    parser.add_argument("--output", type=Path, default=Path("output/decision-data"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    print(json.dumps(write_dataset(args.replays, args.output, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
