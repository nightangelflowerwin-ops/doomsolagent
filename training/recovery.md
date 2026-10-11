# Training interruption and recovery

## October 11, 2026 inspection

The owner reported that the computer went off. Inspection found the existing local trainer still running. Its saved progress advanced from update 213 with 852 examples to update 214 with 856 examples during the check. The operating system reported its last boot on October 6; this inspection did not establish a computer restart or a lost training process.

The trainer remains in `full_corpus_epoch`, consuming all 98,560 training examples before the gameplay image consolidation pass. The existing process was kept running. No duplicate process or fresh training run was started.

The latest saved checkpoint is `update-00000200`, with 200 optimizer updates and 800 consumed examples. Its corpus audit hash matches the current audited corpus. Adapter metadata includes vision and language tensors; optimizer state and random state load successfully. The maximum per parameter optimizer counter is 200. Vision parameters have fewer updates because text only examples do not use the image encoder.

If training stops before another checkpoint is saved, recovery from this checkpoint repeats the examples after cursor 800. A status report or metrics entry is not a replacement for saved model and optimizer state. At update 214, 56 examples had progressed beyond the saved checkpoint.

## Recovery sequence

1. Inspect the current process command and live status. If the existing trainer advances, leave it running.
2. If it has stopped, preserve its status, logs, metrics and checkpoint folders before launching recovery.
3. Read `latest-checkpoint.txt`. Validate the adapter, `state.json`, `optimizer.pt` and `rng.pt`, and require the same corpus audit hash. Reconstruct the same deterministic example order.
4. Resume with `--resume` pointing to the latest complete compatible checkpoint. This restores the adapter, optimizer, random state, phase and cursor. Verify that a new completed update advances beyond that saved cursor.
5. If the latest checkpoint is incomplete, inspect older checkpoints before restarting. If none is usable, preserve the failed run and start a fresh run in a separate output directory. Record the reason and do not describe that run as a continuation.

The local task provides `outputs/Resume-DoomSol-Training.ps1`. It checks whether its recorded training process is already running and otherwise launches the trainer with the saved checkpoint. Launching the script alone does not prove successful recovery; inspect subsequent progress.

For the portable public trainer, use its documented corpus and model arguments plus `--resume /path/to/complete/checkpoint`. Keep the original checkpoint and complete training corpus available. The public trainer publishes completed checkpoints atomically; interrupted temporary directories must not be treated as completed checkpoints.

Training uses the locally downloaded Unsloth base model through the Transformers and PEFT CPU reference backend. Full training, paired holdout evaluation and gameplay validation are unfinished. This inspection establishes continued training and a readable checkpoint, not improved gameplay.
