# CVAT annotation practice: Doom pilot

Stephen Shore annotated enemy tracks in CVAT. This snapshot contains 120 gameplay frames, eight tracks, and 17 explicit keyframes. The 740 exported shape records include interpolation; they are not 740 manual annotations. Labels are draft examples awaiting quality review.

Files: `practice-video.mp4`, `annotations.xml` (CVAT for video 1.1), `labels.json`, and the manifest and transfer verification report. Import the video into a task with the supplied labels, then import the XML using CVAT 1.1.

The source was the Doom region of the existing `jevre-demo-10s-bgm.mp4` composite demonstration. Only frames 0 through 119 are retained. Chess frames 120 through 239 were excluded. This practice clip is separate from the E1M1 mission recording.

## Annotation rules for review

- Use one tight box per enemy, with a stable track identity across frames.
- Mark a track Outside when that enemy disappears; disappearance alone does not mean death.
- Set alive/dead only when visible evidence supports it; otherwise use uncertain.
- Blood effects and player damage are distinct concepts. The supplied schema includes them, but this export contains enemy tracks only.
- Review all tracks, box fit, transitions, and interpolated frames before training.

The original mixed task and private backup are retained locally. Public XML excludes account contact metadata. This commit is a saved snapshot, not a per-drag event log or proof of continuous real-time annotation.

Gameplay and engine assets retain their original attribution and licensing. DoomSol builds on the existing Jevlike work. The separate mission run identifies its producer as MOVINGMAN and its method as Blackwing Decision Method.
