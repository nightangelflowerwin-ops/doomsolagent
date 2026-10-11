"""Seeded input properties; STRESS_SEED changes on every complete suite run."""
import os
import random
import unittest
from agent.archive_data import input_controls, controls


class ControlsStressTests(unittest.TestCase):
    def test_seeded_keyboard_mouse_boundaries(self):
        rng = random.Random(int(os.environ.get('STRESS_SEED', '0')))
        for _ in range(100):
            keys = [key for key in ('KeyA', 'KeyD', 'KeyW', 'KeyS') if rng.randrange(2)]
            dx, dy = rng.uniform(-1000, 1000), rng.uniform(-1000, 1000)
            result = input_controls({'pointerLocked': True, 'heldKeys': keys,
                'events': [{'kind': 'pointer', 'dx': dx, 'dy': dy, 't': 0}]})
            self.assertEqual(result['move_x'], int('KeyD' in keys)-int('KeyA' in keys))
            self.assertEqual(result['move_y'], int('KeyS' in keys)-int('KeyW' in keys))
            self.assertAlmostEqual(result['look_x'], max(-1, min(1, dx/180)), places=6)
            self.assertAlmostEqual(result['look_y'], max(-1, min(1, dy/120)), places=6)
            self.assertEqual(controls(result), result)

    def test_nonfinite_and_out_of_range_axes_rejected(self):
        valid = dict(move_x=0, move_y=0, look_x=0, look_y=0, fire=0, use=0, map=0, escape=0)
        for axis in ('move_x', 'move_y', 'look_x', 'look_y'):
            for bad in (float('nan'), float('inf'), -float('inf'), 1.000001, -1.000001, True, '0'):
                with self.assertRaises(ValueError):
                    controls(dict(valid, **{axis: bad}))
