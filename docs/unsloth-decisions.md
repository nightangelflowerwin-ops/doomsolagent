# Unsloth Decision Training

This experiment learns `none`, `aim`, and `fire` from the browser controller's recorded observations. It uses Unsloth's `FastDecisionModel` format: `state`, `questions`, and `gold`. The existing controller still drives gameplay. The trainer has no bridge connection and cannot send input.

## Prepare Local Data

Capture separate closed-loop replays with the existing agent, then export their `meta.json` files:

```powershell
python agent/decision_data.py replays/run-a/meta.json replays/run-b/meta.json --output output/decision-data
python agent/train_decision_model.py --data output/decision-data --check-data
```

Export needs at least two distinct, nonempty schema 2 closed-loop replays. Whole replays stay together in a deterministic holdout. Copied files are deduplicated by content, and exact observation states in validation are removed from training. This is replay separation, not proof of separate game episodes. Record independent episodes for meaningful generalization tests.

Only pre-decision perception features enter the model input: resolution, normalized aim offsets, candidate confidence and age, and movement evidence. Frame paths, target IDs, controller actions and later validation results are excluded. Invalid observations and unsupported decisions are counted and skipped. Simultaneous button combinations from older ViZDoom traces are not compatible with this browser action schema.

Labels imitate the recorded controller, including dry-run decisions. They are not reviewed optimal actions, confirmed enemies, kills or successful exits. Dataset reports include class counts and missing coverage is shown by `--check-data`.

## Optional Training

Install Unsloth and its model dependencies in a separate environment using its [official installation guide](https://unsloth.ai/docs/get-started/install). The passive agent and dataset exporter do not require Unsloth. Model weights can be large; explicitly choose a model that fits your machine before training. The current local machine has Intel integrated graphics and CPU-only PyTorch, so the bookmarked 4B GPU notebook has not been run here.

```powershell
python agent/train_decision_model.py --data output/decision-data --model path/to/local/model --local-files-only --steps 60
```

Without `--local-files-only`, an explicitly selected Hugging Face model ID can download weights. No model is selected automatically. Training requires every action class in both train and validation before importing Unsloth or loading weights. The trainer rejects skipped/truncated Unsloth inputs, evaluates before and after training on untouched validation rows, and saves locally under `output/`. It does not calibrate on validation or promote a model to gameplay.

The entry point follows the current [Unsloth decision notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_5_(4B)-Decision.ipynb) and was checked against Unsloth source revision `c538b0fb8f576d286e1fa331956885a0d0789aa5`. The full model training path remains untested locally.

## First Data Audit

Four retained local replays produced 233 train rows and 17 validation rows, with 11 rejected decisions. Train contained 126 `none` and 107 `aim`; validation contained 17 `none`. Neither partition contained `fire`. All 250 exported rows passed Unsloth's question/gold schema validation. These counts establish data preparation only. The training gate correctly blocks this dataset.

Next: collect reviewed demonstrations covering all three actions across independent episodes, repeat the audit, then train and compare held-out imitation before considering a bounded shadow-mode evaluation. Visual change after firing does not by itself prove an enemy hit.
