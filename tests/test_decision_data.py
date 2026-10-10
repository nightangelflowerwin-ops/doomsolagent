import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))
from decision_data import export_replay, observation_state, write_dataset
from train_decision_model import check_data, missing_actions


def replay(change=1):
    return {"schema_version": 2, "mode": "closed_loop", "events": [
        {"kind": "observation", "frame": "private/path.jpg", "perception": {
            "resolution": [240, 160], "target": {"id": "secret", "age": 2, "confidence": 0.7},
            "aim_error": {"dx": 20, "dy": 10, "distance": 22.36},
            "movement": {"state": "turning", "global_change": change, "center_change": 9},
        }},
        {"kind": "decision", "action": {"type": "aim", "dx": 4, "target_id": "secret"}},
        {"kind": "validation", "validation": {"result": "confirmed", "kills": 999}},
    ]}


class DecisionDataTests(unittest.TestCase):
    def test_pre_action_features_do_not_include_label_or_outcome(self):
        rows, skipped = export_replay(replay())
        self.assertEqual(skipped, 0)
        self.assertEqual(rows[0]["gold"], {"action": "aim"})
        state = rows[0]["state"]
        for forbidden in ("confirmed", "kills", "secret", "private", "action", "frame"):
            self.assertNotIn(forbidden, state)
        self.assertAlmostEqual(json.loads(state)["aim_error"]["dx"], 20 / 240, places=5)

    def test_decisions_cannot_reuse_stale_observation(self):
        meta = replay()
        meta["events"].append({"kind": "decision", "action": {"type": "fire"}})
        rows, skipped = export_replay(meta)
        self.assertEqual((len(rows), skipped), (1, 1))
        meta["events"].insert(1, {"kind": "action", "sent": True})
        self.assertEqual(export_replay(meta), ([], 2))

    def test_invalid_observations_and_unknown_actions_are_rejected(self):
        for mutate in (
            lambda m: m["events"][0]["perception"]["resolution"].__setitem__(0, 0),
            lambda m: m["events"][0]["perception"]["target"].__setitem__("confidence", float("nan")),
            lambda m: m["events"][1]["action"].__setitem__("type", "jump"),
        ):
            meta = replay(); mutate(meta)
            self.assertEqual(export_replay(meta), ([], 1))

    def test_no_target_observation_is_supported(self):
        perception = replay()["events"][0]["perception"]
        perception.update(target=None, aim_error=None)
        self.assertIsNone(json.loads(observation_state(perception))["target"])

    def test_replay_split_is_deterministic_and_copies_stay_together(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root); paths = []
            for index, meta in enumerate((replay(1), replay(2), replay(3), replay(1))):
                path = root / f"{index}.json"; path.write_text(json.dumps(meta)); paths.append(path)
            report = write_dataset(paths, root / "a")
            other = write_dataset(list(reversed(paths)), root / "b")
            self.assertEqual(report, other)
            self.assertEqual(sum(map(len, report["groups"].values())), 3)
            train, validation = check_data(root / "a")
            self.assertEqual(len(train) + len(validation), 3)
            row = copy.deepcopy(validation[0])
            with (root / "a/train.jsonl").open("a") as handle:
                handle.write(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "overlap"):
                check_data(root / "a")

    def test_single_replay_cannot_claim_independent_holdout(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "meta.json"; path.write_text(json.dumps(replay()))
            with self.assertRaisesRegex(ValueError, "two distinct"):
                write_dataset([path, path], Path(root) / "out")

    def test_missing_action_coverage_is_visible_before_model_loading(self):
        rows, _ = export_replay(replay())
        self.assertEqual(missing_actions(rows), ["fire", "none"])


if __name__ == "__main__":
    unittest.main()
