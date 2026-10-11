import tempfile
import unittest
from pathlib import Path
from agent.training_checkpoint import publish_checkpoint

class CheckpointTests(unittest.TestCase):
    def test_success_publishes_files_and_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'checkpoints'
            folder=publish_checkpoint(root,'update-1',lambda p:(p/'state.json').write_text('{}'))
            self.assertTrue((folder/'state.json').is_file())
            self.assertEqual((Path(tmp)/'latest-checkpoint.txt').read_text(),str(folder.resolve()))

    def test_interrupted_save_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'checkpoints'
            old=publish_checkpoint(root,'update-1',lambda p:(p/'state.json').write_text('{}'))
            def fail(path):
                (path/'partial').write_text('unfinished')
                raise RuntimeError('simulated interrupted save')
            with self.assertRaisesRegex(RuntimeError,'interrupted'): publish_checkpoint(root,'update-2',fail)
            self.assertFalse((root/'update-2').exists())
            self.assertEqual((Path(tmp)/'latest-checkpoint.txt').read_text(),str(old.resolve()))

    def test_existing_checkpoint_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'checkpoints'
            publish_checkpoint(root,'update-1',lambda p:(p/'state.json').write_text('original'))
            with self.assertRaises(FileExistsError): publish_checkpoint(root,'update-1',lambda p:None)
            self.assertEqual((root/'update-1/state.json').read_text(),'original')
