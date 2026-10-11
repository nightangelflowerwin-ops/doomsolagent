"""Portable published corpus reader with strict paths and content verification."""
import hashlib
import json
from pathlib import Path
from PIL import Image

DOMAINS = {'engine_observation_and_map_memory', 'doomsol_gameplay_image',
           'browser_candidate_observation', 'vizdoom_policy_gameplay_image', 'dwasm_hud_grounding'}

def image_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not relative.startswith('images/'):
        raise ValueError('Image path escapes the dataset image directory')
    return path

def records(root, split, verify_images=True):
    if split not in {'train', 'validation'}:
        raise ValueError('Unknown split')
    seen = set()
    with (Path(root) / f'{split}.jsonl').open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if row['split'] != split or row['domain'] not in DOMAINS:
                raise ValueError('Invalid split or domain')
            if row['id'] in seen:
                raise ValueError('Duplicate example id')
            seen.add(row['id'])
            if not isinstance(row['target'], dict) or not isinstance(row['state'], dict):
                raise ValueError('Invalid state or target')
            if row.get('image'):
                path = image_path(root, row['image'])
                if verify_images:
                    if hashlib.sha256(path.read_bytes()).hexdigest() != row['image_sha256']:
                        raise ValueError('Image hash mismatch')
                    with Image.open(path) as image:
                        image.verify()
            yield row

def validate(root):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    counts = {}
    for split in ('train', 'validation'):
        if hashlib.sha256((root / f'{split}.jsonl').read_bytes()).hexdigest() != manifest[f'{split}_sha256']:
            raise ValueError('Split file hash mismatch')
        for row in records(root, split):
            key = f"{split}:{row['domain']}"
            counts[key] = counts.get(key, 0) + 1
    if counts != manifest['counts'] or sum(counts.values()) != manifest['examples']:
        raise ValueError('Manifest count mismatch')
    return counts

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path)
    print(json.dumps(validate(parser.parse_args().dataset), indent=2))
