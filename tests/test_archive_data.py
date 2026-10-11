import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image
from agent.archive_data import build_archive, conversation, input_controls, safe_frame
from agent.train_archive_model import check_archive


class ArchiveTests(unittest.TestCase):
    def test_legacy_test_blocks_and_boundary_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); run = root / "replays" / "run"; run.mkdir(parents=True)
            model = root / "models" / "keyboard"; model.mkdir(parents=True)
            (run / "meta.json").write_text(json.dumps({"source_url": "https://doomsol.io", "mode": "human_demonstration"}))
            rows = []
            for index, t in enumerate((0, 19500, 20100, 21000)):
                Image.new("RGB", (160, 100), (index * 50, 0, 0)).save(run / f"{index}.png")
                rows.append({"run": "run", "frame": f"{index}.png", "wall_time": t,
                    "move_x": 0, "move_y": 0, "look_x": 0, "look_y": 0,
                    "fire": 0, "use": 0, "map": 0, "escape": 0, "pointer_locked": True})
            (model / "train_sequences.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            (model / "test_sequences.jsonl").write_text(json.dumps(rows[2]))
            report = build_archive(root, root / "out")
            self.assertEqual(report["train"]["rows"], 1)
            self.assertEqual(report["validation"]["rows"], 2)
            self.assertEqual(report["counts"]["holdout_boundary_guard"], 1)
            self.assertEqual(report["train"]["groups"], ["run:0"])
            self.assertEqual(report["validation"]["groups"], ["run:1"])
            check_archive(root / "out")

    def test_controls_and_long_batches(self):
        sample = {"pointerLocked": True, "heldKeys": ["KeyW"], "events": [
            {"kind": "pointer", "dx": 360, "dy": -120, "t": 1},
            {"kind": "button", "button": 0, "down": 1, "t": 2}]}
        action = input_controls(sample)
        self.assertEqual((action["move_y"], action["look_x"], action["look_y"], action["fire"]), (-1, 1, -1, 1))
        sample["events"][-1]["t"] = 1000
        with self.assertRaises(ValueError):
            input_controls(sample)

    def test_provenance_holdout_duplicates_and_no_metadata_input(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); models = root / "models" / "keyboard"; models.mkdir(parents=True)
            rows = []
            for name, host, color in (("a", "doomsol.com", "red"), ("b", "doomsol.io", "blue"), ("other", "example.com", "green")):
                run = root / "replays" / name; run.mkdir(parents=True)
                Image.new("RGB", (160, 100), color).save(run / "f.png")
                (run / "meta.json").write_text(json.dumps({"source_url": f"https://{host}", "game_ids": ["doomsol"],
                    "mode": "human_demonstration", "frames": []}), encoding="utf-8")
                rows.append({"run": name, "frame": "f.png", "wall_time": 100,
                             "move_x": 0, "move_y": -1, "look_x": .1, "look_y": 0,
                             "fire": 1, "use": 0, "map": 0, "escape": 0, "pointer_locked": True,
                             "kills": 99, "death_score": 1})
            (models / "train_sequences.jsonl").write_text("\n".join(json.dumps(r) for r in rows + rows), encoding="utf-8")
            (models / "test_sequences.jsonl").write_text(json.dumps(rows[1]), encoding="utf-8")
            report = build_archive(root, root / "out")
            self.assertEqual(report["train"]["recordings"], ["a"])
            self.assertEqual(report["validation"]["recordings"], ["b"])
            self.assertEqual(report["train"]["rows"], 1)
            self.assertEqual(report["validation"]["rows"], 1)
            self.assertEqual(len(check_archive(root / "out")["train"]), 1)
            row = json.loads((root / "out" / "train.jsonl").read_text())
            messages = conversation(row, root / "out")["messages"]
            self.assertEqual(len(messages[0]["content"]), 2)
            self.assertNotIn("kills", str(messages))
            self.assertNotIn("recording", str(messages))
            with self.assertRaises(ValueError):
                safe_frame(root, "../outside.png")
            with self.assertRaises(ValueError):
                build_archive(root, root / "out")
            # Changing exported pixels must fail before any model is loaded.
            Image.new("RGB", (384, 224), "black").save(root / "out" / row["image"])
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                check_archive(root / "out")


if __name__ == "__main__":
    unittest.main()
