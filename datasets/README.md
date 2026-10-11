# DoomSol training data

The complete final training corpus is published in the [training-data-v1 release](https://github.com/nightangelflowerwin-ops/doomsolagent/releases/tag/training-data-v1). Download every dataset ZIP part and extract them into the same directory. The release includes the training examples, validation examples, original referenced gameplay images, source attribution, and SHA-256 manifest. Local machine paths are removed.

There are **98,560 training examples and 61,330 validation examples**:

| Task | Train | Validation | Meaning |
| --- | ---: | ---: | --- |
| Engine observation and map memory | 85,225 | 58,566 | Privileged engine observations and recorded controller actions |
| Human DoomSol gameplay controls | 3,311 | 800 | Gameplay viewport images and continuous keyboard/mouse controls |
| ViZDoom policy gameplay images | 1,629 | 1,810 | Real pre-action images and older policy actions |
| Rendered HUD grounding | 8,211 | 140 | Synchronized rendered images and health/armor labels |
| Browser candidate observations | 184 | 14 | Structured perception and recorded candidate decisions |

Every JSONL record contains `id`, `split`, `domain`, `state`, `target`, `provenance`, and optional `image` and `image_sha256`. Image paths are relative to the dataset directory. The manifest lists file hashes, image sizes, decoded-pixel hashes, counts, and the original corpus audit hash. Source signatures and attribution remain in provenance.

Validate the downloaded data before training:

```sh
python -m agent.training_dataset /path/to/extracted/dataset
```

Train the complete corpus locally with the downloaded Unsloth Qwen3.5-0.8B base model:

```sh
python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-training-cpu.txt
python -m agent.train_corpus_model --dataset /path/to/extracted/dataset --model /path/to/Qwen3.5-0.8B --output /path/to/run
```

The CPU reference backend uses Transformers and PEFT, with rank-8 adapters on vision and language linear layers. It consumes every training example once and then every gameplay-action image once in a consolidation pass. It has no maximum-step cutoff. Only assistant tokens contribute to loss; oversized records raise an error instead of being truncated. Checkpoints retain adapter, optimizer, random state, and corpus identity. Resume with `--resume /path/to/checkpoint`.

After training, the trainer evaluates every held-out example with the base model and trained adapter, recording token loss, generated JSON, exact actions, continuous-control error, and button confusion counts. This can take substantial time on a CPU.

## Interpretation and limitations

These are demonstrations and observation labels, not optimal-policy labels. Older controllers include unsuccessful actions. Engine observations contain privileged map information and must not be treated as camera-only inputs. The HUD task reads displayed values; it does not predict next actions. Older teacher records with unsynchronized state and images were excluded rather than assigned invented labels. Duplicate examples were removed. Decoded image content does not overlap training and validation. Human holdout blocks still share recording environments, and independence of all historical campaign seeds is not established.

The release contains every example in the final eligible corpus. Rejected, corrupted, duplicated, unsynchronized, and unsupported historical records are described in the source audit and are not presented as training examples. Full raw campaign exports are not substituted for the verified corpus.

Offline tests and imitation loss do not establish better gameplay. Promotion requires actual engine gameplay and genuine exit evidence against the current controller.

Repeat the complete offline suite 1,000 times, preserving seeds, results, and source hashes:

```sh
python tools/verify_1000.py --output /path/to/fresh/verification-directory
```

## Attribution and reuse

Original annotation and controller trace contributions are shared for reuse under CC BY 4.0; retain the producer attribution and provenance. This grant covers those contributions, not third-party game assets. Rendered Freedoom assets retain the Freedoom modified BSD license, included as `FREEDOOM-COPYING.txt`. Browser gameplay images retain the rights of their underlying game assets and their source host attribution; no ownership of third-party assets is asserted. The data card and notices travel with every dataset package.

The base model is separately distributed by [Unsloth](https://huggingface.co/unsloth/Qwen3.5-0.8B); it is not included in this release.
