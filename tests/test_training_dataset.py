import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from agent.training_dataset import image_path, records

class PublishedDatasetTests(unittest.TestCase):
    def test_escape_paths_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for path in ('../private.png', 'images/../../private.png', 'other/frame.png'):
                with self.assertRaises(ValueError): image_path(tmp, path)

    def test_roundtrip_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'images').mkdir()
            path=root/'images/frame.png'; Image.new('RGB',(4,4),'red').save(path)
            row=dict(id=1, split='train',domain='doomsol_gameplay_image',state={},target={'fire':0},
                     image='images/frame.png',image_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            (root/'train.jsonl').write_text(json.dumps(row)+'\n')
            self.assertEqual(list(records(root,'train')), [row])
            Image.new('RGB',(4,4),'blue').save(path)
            with self.assertRaisesRegex(ValueError,'hash mismatch'): list(records(root,'train'))

    def test_duplicate_ids_and_invalid_split_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            row=dict(id=1,split='train',domain='browser_candidate_observation',state={},target={})
            (root/'train.jsonl').write_text((json.dumps(row)+'\n')*2)
            with self.assertRaisesRegex(ValueError,'Duplicate'): list(records(root,'train'))
            with self.assertRaisesRegex(ValueError,'Unknown split'): list(records(root,'test'))
