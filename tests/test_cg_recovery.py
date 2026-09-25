"""CPU recovery tests; run with python -m unittest discover -s tests -v."""
import ast
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from EasyLM import cg_resume as cg
from sweep_launcher import selected_value


TRAINER = (Path(__file__).resolve().parents[1]
           / 'EasyLM/models/llama/llama_train_gn.py')


def trainer_function(name):
    tree = ast.parse(TRAINER.read_text())
    node = next(item for item in tree.body
                if isinstance(item, ast.FunctionDef) and item.name == name)
    namespace = {}
    exec(compile(ast.fix_missing_locations(
        ast.Module(body=[node], type_ignores=[])), str(TRAINER), 'exec'), namespace)
    return namespace[name]


def snapshot(step):
    return (dict(step=step, data_consumption_version=cg.DATA_CONSUMPTION_VERSION,
                 training_progress=dict(phase_completed_updates=step,
                                        dataset_total_tokens=step * 32)),
            dict(packed_state_version=1, metadata=dict(dataset_total_tokens=step * 32)))


def save(directory, generation, step):
    metadata, dataset = snapshot(step)
    return cg.save_bundle(str(directory), generation, metadata, dataset,
                          lambda path: Path(path).write_bytes(f'state-{step}'.encode()))


class RecoveryTests(unittest.TestCase):
    def test_condition_diagnostic_solver_support(self):
        supported = trainer_function('supports_condition_diagnostics')
        self.assertTrue(supported('cg', False))
        self.assertTrue(supported('adamw', True))
        self.assertTrue(supported('muon', True))
        self.assertFalse(supported('adamw', False))
        self.assertFalse(supported('muon', False))
        self.assertFalse(supported('unknown', True))

    def test_spectrum_flags_are_reporting_only(self):
        saved = {'optimizer_type': 'cg', 'condition_log': False}
        current = dict(saved, condition_log=True,
                       spectrum_endpoint_maxiter=48,
                       spectrum_check_every=8)
        cg.validate_flags(saved, current)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_lambda(self):
        for batch, expected in ((256, .025), (1024, .1), (2048, .2)):
            self.assertEqual(cg.batch_lambda(batch, 10240), expected)
        cg.validate_batch_lambda(0, 'muon', -1, 0, 2048)
        for args in ((-1, 'cg', -1, 0, 256),
                     (float('nan'), 'cg', -1, 0, 256),
                     (float('inf'), 'cg', -1, 0, 256),
                     (1024, 'muon', -1, 0, 256),
                     (1024, 'cg', .4, 10, 256),
                     (1024, 'cg', -1, 0, 2048)):
            with self.assertRaises(ValueError):
                cg.validate_batch_lambda(*args)
        with self.assertRaises(ValueError):
            cg.batch_lambda(2048, 1024)

    def test_rolling_fallback_and_retained_milestones(self):
        gen = save(self.root, -1, 1)
        cg.retain_milestone(str(self.root), gen, 1)
        milestone = self.root / 'milestones/step_1/cg_state_0'
        original = milestone.read_bytes()
        for step in (2, 3, 4):
            gen = save(self.root, gen, step)
            cg.retain_milestone(str(self.root), gen, step)
        self.assertEqual(gen, 3)
        self.assertEqual(milestone.read_bytes(), original)
        # An already valid milestone is never replaced by later state.
        cg.retain_milestone(str(self.root), gen, 1)
        self.assertEqual(milestone.read_bytes(), original)
        self.assertEqual(cg.read_bundle(str(milestone))[0]['step'], 1)
        (self.root / 'cg_state_1').write_bytes(b'corrupt')
        self.assertEqual(cg.read_bundle(cg.newest(str(self.root)))[0]['step'], 3)
        (self.root / 'cg_state_0').write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'No valid'):
            cg.resolve_resume(str(self.root), exists=True)
        self.assertEqual(cg.resolve_resume(str(self.root), str(milestone), exists=True), str(milestone))

    def test_interrupted_milestone_retry(self):
        gen = save(self.root, -1, 5)
        original = cg.durable_write
        def fail_marker(path, data):
            if 'milestones' in path and b'"complete": true' in data:
                raise OSError('interrupted')
            return original(path, data)
        with patch.object(cg, 'durable_write', side_effect=fail_marker):
            with self.assertRaises(OSError):
                cg.retain_milestone(str(self.root), gen, 5)
        folder = self.root / 'milestones/step_5'
        self.assertIsNone(cg.newest(str(folder)))
        self.assertIsNotNone(cg.newest(str(self.root)))
        cg.retain_milestone(str(self.root), gen, 5)
        self.assertIsNotNone(cg.newest(str(folder)))

    def test_resolution_and_version(self):
        self.assertIsNone(cg.resolve_resume(str(self.root)))
        with self.assertRaises(ValueError):
            cg.resolve_resume(str(self.root), requested=True)
        with self.assertRaises(ValueError):
            cg.resolve_resume(str(self.root), exists=True)
        save(self.root, -1, 2)
        valid = str(self.root / 'cg_state_0')
        with self.assertRaises(FileNotFoundError):
            cg.resolve_resume(str(self.root), str(self.root / 'missing/cg_state_0'), exists=True)
        self.assertEqual(cg.resolve_resume(str(self.root), valid), valid)
        with self.assertRaisesRegex(ValueError, 'old discarded-fetch'):
            cg.validate_consumption({})
        with self.assertRaisesRegex(ValueError, 'trajectory'):
            cg.validate_flags({'cg_lambda_batch_denominator': 10240},
                              {'cg_lambda_batch_denominator': 20480})

    def test_permissions_are_not_missing_checkpoints(self):
        with patch.object(cg, 'reader', side_effect=PermissionError('denied')):
            with self.assertRaises(PermissionError):
                cg.newest(str(self.root))

    def test_nonwriter_does_not_touch_storage(self):
        calls = []
        metadata, dataset = snapshot(1)
        self.assertEqual(cg.save_bundle(str(self.root), 2, metadata, dataset,
                                       calls.append, enable=False), 2)
        cg.retain_milestone(str(self.root), 2, 1, enable=False)
        self.assertEqual(calls, ['/dev/null'])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_selected_sweep_values(self):
        self.assertEqual(selected_value({'--cg_resume_state': 'selected'},
                                        ['--cg_resume_state=static'], 'cg_resume_state'), 'selected')

    def test_mock_gcs_roundtrip_and_copy(self):
        objects = {}
        class Blob:
            def __init__(self, path):
                self.name = path
                self.bucket = self
            def open(self, mode):
                if self.name not in objects:
                    raise FileNotFoundError(self.name)
                return io.BytesIO(objects[self.name])
            def upload_from_string(self, data):
                objects[self.name] = data
            def upload_from_filename(self, path):
                objects[self.name] = Path(path).read_bytes()
            def copy_blob(self, source, bucket, new_name):
                objects[new_name] = objects[source.name]
        with patch.object(cg, 'blob', side_effect=Blob):
            directory = 'gs://test/run'
            gen = save(directory, -1, 3)
            cg.retain_milestone(directory, gen, 3)
            selected = cg.resolve_resume(directory, exists=True)
            self.assertEqual(cg.read_bundle(selected)[0]['step'], 3)
            self.assertEqual(cg.read_bundle(directory + '/milestones/step_3/cg_state_0')[0]['step'], 3)
            objects[directory + '/cg_state_0'] = b'corrupt'
            self.assertIsNone(cg.newest(directory))


if __name__ == '__main__':
    unittest.main()
