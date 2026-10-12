# Nightangel overnight training watchdog

Built October 11, 2026.

## Purpose

Keep the local DoomSol training run working overnight without requiring the owner to watch the process. The tool prevents Windows idle sleep while it runs, checks for the trainer every minute, and restarts an exited trainer from the latest complete checkpoint compatible with the audited corpus.

This is a Windows training operations tool. It does not train a separate model, improve model quality by itself, or replace validation and gameplay testing.

## What was built

- [Watchdog source](../tools/training_watchdog.py), using Python standard libraries and Windows interfaces.
- Detection of an existing trainer to avoid duplicate launches, with a file lock preventing duplicate watchdogs.
- Checkpoint selection that skips incomplete folders and requires the original corpus audit hash.
- Seven thread recovery launches and a five minute interval between restart attempts.
- A Windows idle sleep prevention request that is released when the watchdog exits.
- A local status report recording monitoring, recovery attempts and inspection errors.

## Local deployment

The installed local copy runs independently of this chat. A current user Windows sign-in startup entry starts it after sign-in. Task Scheduler registration was denied, so the deployment uses a windowless background process and the sign-in entry instead. It does not start before sign-in after a reboot.

The public source requires an already configured DoomSol training workspace:

```powershell
python tools/training_watchdog.py --task-root C:/path/to/training/workspace
```

The workspace must contain `work/train_faithful_model.py`, `work/training-env/Scripts/python.exe`, `work/faithful-corpus-v6/audit.json`, `work/faithful-training-v6/checkpoints`, and `outputs/local-training-status.json`. The local trainer must support `--resume` and `--threads`. This helper targets that existing workspace layout; it is not a general trainer installer. The root defaults are never taken from a contributor computer path.

Reports appear in `outputs/Training-Watchdog.json`. Recovery logs stay in the local `work` folder. Creating `outputs/Stop-Training-Watchdog.flag` stops the watchdog on its next check. Remove its Windows sign-in startup entry as well when disabling it permanently. Stopping the watchdog leaves the trainer running.

## Verification and limits

Verified during deployment: the watchdog detects the existing trainer without launching a duplicate, the Windows sleep API accepted its request, an incomplete newest checkpoint falls back to an older complete folder, and a mismatched corpus is rejected. The local trainer continued advancing after installation. A trainer crash was not intentionally induced as a test.

Keep the laptop plugged in with the lid open. The tool prevents idle sleep, not manual sleep, lid closure, power failure or forced shutdown. It detects an exited process; it does not currently diagnose or kill a process that is still present but stalled. Recovery replays examples processed after the last saved checkpoint. If no complete compatible checkpoint is found, it reports an inspection error and preserves the run rather than overwriting it with a fresh model.
