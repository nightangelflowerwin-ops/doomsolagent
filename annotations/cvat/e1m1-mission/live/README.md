# Automatic saved annotation checkpoints

`latest.json` contains CVAT's native annotation structure and the corresponding label IDs and attribute specifications. `checkpoints.json` records observation time, CVAT save time, a content hash, and changed-record counts. Counts include track metadata, keyframes, and Outside markers. The first checkpoint is a baseline, not newly completed work.

The local watcher checks every 30 seconds and commits only when saved annotations or labels change. It retries failed pushes before accepting another checkpoint. Git history preserves earlier snapshots. Unsaved browser edits are not captured. This is saved-state syncing, not a log of every mouse movement.

Timestamps and differences provide an auditable work trail. They do not establish continuous working time, manual review of every frame, human authorship, or annotation accuracy. Accuracy requires independent review or comparison against adjudicated labels. Automated GitHub contributions are not independent assessment scores.

Run `python tools/sync_cvat.py` from a clean, dedicated clone with existing GitHub credentials and the local CVAT Docker installation running in WSL Ubuntu. Stop the watcher to work manually in that clone. No credentials, account records, or contact information are exported.
