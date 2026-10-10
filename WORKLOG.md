# Build Log

This file is the public record of ongoing work. Each meaningful coding, data, training, or evaluation session should add one short entry with evidence and the next question.

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
