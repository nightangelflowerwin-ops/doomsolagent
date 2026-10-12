# Build Log

This file is the public record of ongoing work. Each meaningful coding, data, training, or evaluation session should add one short entry with evidence and the next question.

## 2026-10-11 Training continuity inspection

The owner reported a computer shutdown. The existing local trainer was still active and advanced from 213 updates and 852 examples to 214 updates and 856 examples during inspection. Validated the saved checkpoint at update 200 and cursor 800 against the unchanged corpus audit, optimizer and random state, and vision/language adapter metadata. Kept the advancing process running. Added [recovery instructions](training/recovery.md) describing the saved sequence, replay after interruption and fallback to earlier complete checkpoints before a fresh run. Training and gameplay validation remain unfinished.

## 2026-10-10 Nightangel contribution reporting

The working branch is named `nightangelflowerwin`. Added GitHub verification runs and a local reporter for meaningful daily training progress and milestones on `main`, attributed to the owner account. Reports exclude machine details and unchanged progress does not produce extra commits. The expanded suite has 26 tests; complete repeated verification is recorded in GitHub Actions. Model training remains local and gameplay validation remains unfinished.

## 2026-10-10 - Complete reusable corpus and repeated verification

- Changed: publish a portable corpus reader, complete CPU vision/language LoRA trainer, checkpoint publication guard, dataset section, and repeatable 1,000-run verifier. Export every final eligible example and original referenced image with relative paths, source attribution, hashes, and preserved validation splits.
- Data: 98,560 training and 61,330 validation examples across five explicitly separated tasks. All exported records were parsed and every referenced image decoded and hashed; decoded images do not overlap the two splits.
- Verification: the expanded 24-test suite completed 1,000 runs against frozen source hashes: 24,000 executions, zero failures. A separate 1,000-round check completed 5,000 checks with the actual local Qwen processor across all five domains. Complete result ledgers accompany the release. The public reader validated every exported example and image, and every target was checked against its task schema.
- Training: the local Unsloth Qwen3.5-0.8B base is training vision and language adapters through the Transformers/PEFT CPU reference backend. Training consumes a complete corpus pass and complete gameplay-image consolidation pass, with no step cap.
- Limitation: historical actions include unsuccessful controllers; engine observations are privileged data; HUD grounding is not action prediction. Training and full paired holdout evaluation are unfinished. No gameplay improvement or model promotion is claimed.
- Next: finish the complete training and holdout evaluation, then compare real engine gameplay against the current controller using fixed seeds and genuine exit evidence.

## 2026-10-10 - Broader Human Training Archive

- Changed: inventory archived model datasets and raw human inputs; export verified DoomSol keyboard recordings as cropped image/control conversations for optional Unsloth vision training. Preserve historical evaluation blocks, guard boundaries by 850 ms, remove exact duplicate examples and train/validation image overlap.
- Evidence: 12 tests pass. The local export contains 3,311 training examples (702 firing) and 800 validation examples (65 firing); 486 duplicate examples, 67 boundary examples and four shared training images were removed.
- Limitation: 7,195 derived rows have missing DoomSol provenance or belong to another game; simulator manifests are inventoried but excluded. Some captured sessions have no input rows. Main validation shares a recording with training and measures approximate synchronized imitation, not episode generalization. Model training has not run locally.
- Next: use the optional vision trainer with a compatible local model and compare held-out loss before any shadow gameplay evaluation.

## 2026-10-10 - Unsloth Decision Data Bridge

- Changed: export pre-decision browser observations to Unsloth's choice-model format; keep whole replays in separate partitions; remove exact state overlap; add an optional training entry point with class-coverage gates.
- Evidence: nine focused tests pass. Four retained local replays yield 233 train and 17 validation decisions, with 11 rejected decisions; all exported rows pass upstream question and gold-label validation.
- Limitation: the existing data lacks fire demonstrations and validation aim examples. Training is blocked by coverage checks; no model was trained and no gameplay improvement is claimed.
- Next: record reviewed demonstrations across independent episodes with all three action classes, then train and measure held-out imitation.

## 2026 08 27   Public verification workflow

- Changed: added a GitHub Actions workflow and a documented end-of-session routine.
- Evidence: published the project and configured each pushed change to install its image dependency, run the focused perception tests, and confirm the public project records are present.
- Limitation: the current tests cover the perception layer, not end-to-end gameplay.
- Next: add an independent replay fixture and an episode-level evaluation check.

## 2026 08 22   Teacher and recovery evaluation artifacts

- Retained deterministic teacher, recovery-gate, lifecycle, and five-minute admission reports for repeatable evaluation.
- Continued separating trainable observations from inconclusive or unsafe action evidence.
- Next: consolidate the strongest evaluation path into a reproducible command and record another independent episode.

## 2026 08 16   Imitation policy v4 experiment

- Trained `doom_policy_v4` on 87 sequences and evaluated against 18 time-block validation sequences.
- Best validation loss was 0.5767 at epoch 9.
- Recorded the central limitation: one source episode is not enough to establish episode-level generalization.
- Next: capture more independent demonstrations and compare episode-held-out performance.

## 2026 08 15   Closed loop observation milestone

- Added passive discovery and scouting modes.
- Added candidate tracking, crosshair-relative aim error, and structured replay events.
- Kept live action bounded and opt-in.
- Next: improve enemy evidence quality before treating firing results as conclusive.

## Entry template

```text
## YYYY-MM-DD - Short milestone

- Changed:
- Evidence:
- Limitation:
- Next:
```
# 2026-10-10: Complete vision passes and a CPU reference backend

The archive trainer now defaults to a complete epoch, with an explicit debug-only step cap. CPU machines use Transformers/PEFT adapters in both the vision and language linear layers; compatible GPU machines retain the Unsloth backend. Real images and continuous control labels are preserved, and the CPU loss covers only the assistant action response. Checkpoints support resuming training.

Validation: all 12 repository tests pass; the archive's 3,311 training images and 800 held-out images pass content-hash, control-schema and overlap checks. A local Qwen3.5-0.8B vision forward pass, backward pass and saved adapter succeeded. Complete corpus training and gameplay improvement remain unverified. The broader local campaign and embedded-frame training audit remains outside the public repository.

## 2026-10-11: CVAT annotation examples

Published a draft Doom-only CVAT snapshot: 120 frames, eight enemy tracks, 17 explicit keyframes, and 740 exported records including interpolation. Removed chess frames and public account contact metadata. Verified retained frame bounds, track count, attributes, and keyframes against the local imported task. Quality review remains pending. Selected the archived E1M1 switch-aware-47 recording for the next annotation task; its run report records one map completed, not the full episode.
Validation: dataset structure and contact-metadata checks passed. The existing test suite ran 23 tests: 22 passed; corpus-training test import was blocked because torch is absent from this Python environment.
Saved mission annotation checkpoint: 22 tracks, 63 keyframes, latest keyframe 202. Preserved the saved database snapshot for audit; quality review and a portable CVAT export remain pending.
