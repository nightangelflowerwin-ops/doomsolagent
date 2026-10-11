# Nightangel training progress

Training runs on the owner computer. `progress.json` records the latest published progress without computer paths, process IDs, or credentials. Its commit history preserves previous reports.

The local reporter publishes a commit under the configured human author when training advances on a new UTC date, crosses another 10,000 examples, changes phase, or changes stage. Unchanged status and individual optimizer steps do not produce repeated commits. GitHub verifies the author against the authenticated account.

GitHub Actions records 1,000 complete offline suite runs on daily verification, manual verification, and a new published progress report. Results are downloadable workflow artifacts. These runs verify the project; they do not move model training onto GitHub hardware or prove gameplay improvement.

Start the reporter locally after configuring Git credentials and the author associated with your GitHub account:

```sh
python tools/publish_training_progress.py --status /path/to/local/training/status.json --watch
```

The reporter uses the existing Git credential helper. It never saves or prints the credential. Reports go to `main`, where eligible human authored commits can appear on the contribution graph.

Read [training interruption and recovery](recovery.md) for the October 11 inspection, last recoverable sequence and resume procedure.
