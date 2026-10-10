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

Visual change after firing does not by itself prove an enemy hit.

## Use the Existing Human Archive

The broader archive needs a different task. Keyboard demonstrations contain simultaneous movement, mouse offsets, firing, interaction, map and escape controls. Turning is not necessarily aiming at a candidate. `archive_data.py` therefore preserves these controls for image-to-JSON imitation, using Unsloth's [vision conversation format](https://unsloth.ai/docs/basics/vision-fine-tuning). It does not relabel them as the three controller decisions above.

```powershell
python agent/archive_data.py path/to/local/archive --output output/archive-vision
python agent/train_archive_model.py --data output/archive-vision --check-data
python agent/train_archive_model.py --data output/archive-vision --model path/to/local/vision-model --local-files-only --steps 60
```

The archive directory contains `models/` and `replays/`. Export scans every model JSONL and raw human input file, records row counts and exclusion reasons in `report.json`, and accepts only human recordings with explicit DoomSol host provenance. It supports the historical `.com` host and `.io`. Missing provenance, other games, simulator teacher manifests and unsupported control schemas are excluded rather than silently mixed into the task. Checkpoint files are learned weights, not demonstrations. The four closed-loop replays remain available through the separate decision exporter.

Historical test frames reserve their 20-second time blocks for validation, even when later model folders copied them into training. Training excludes another 850 ms at holdout boundaries. Without historical test blocks, a deterministic whole-recording holdout is used. Exact duplicate image/control examples are collapsed, and image content shared with validation is removed from training. Source frame paths must remain inside their replay. Cropped gameplay images, labels and source metadata are local ignored artifacts. Only the image and static instructions enter the vision model input; controls appear solely as the assistant response. Outcomes, kills, timestamps, recording IDs and filesystem paths stay out of model inputs.

Raw keyboard batches use a preceding frame within 850 ms; unlocked recordings and event batches spanning more than 850 ms are rejected. Legacy synchronized sequence labels retain their original alignment and temporal fire dilation, so the dataset is approximate behavioral imitation, not proof of causal next-action timing. All recorded controls are retained without active-only filtering. Whole recordings can still be parts of the same episode, and single images omit motion history. The fixed legacy viewport crop should be reviewed when changing capture layout.

The optional vision trainer checks image hashes, holdout group separation and firing/non-firing coverage before loading a chosen model. It evaluates held-out imitation loss before and after training and saves the adapter and processor locally. This is separate from `FastDecisionModel`, requires compatible Unsloth vision dependencies and hardware, and has not been model-trained locally. Neither trainer promotes a model into live gameplay.

The local archive export yielded 3,311 training examples with 702 firing labels and 800 validation examples with 65 firing labels. It removed 486 duplicate image/control examples, 67 examples near holdout boundaries and four training examples sharing validation pixels. Another 7,195 derived rows had missing DoomSol provenance or came from another game. Simulator manifest counts and empty input files are visible in the local inventory. These are prepared demonstration counts, not trained-model results.
