"""Exercise the launcher through command construction without launching training."""
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import sweep_launcher
from test_cg_recovery import save


class LauncherTests(unittest.TestCase):
    def launch(self, root, extra=()):
        args = ['sweep_launcher.py', '--program=EasyLM.models.llama.llama_train_gn',
                '--optimizer_type=cg', '--train_dataset_batch_size=256',
                '--output_dir=' + root, *extra]
        fake_gcs = types.ModuleType('EasyLM.gcs_utils')
        fake_gcs.gcs_path_exists = lambda path: False
        fake_gcs.read_from_gcs = lambda path: 'id'
        with patch.dict(sys.modules, {'EasyLM.gcs_utils': fake_gcs}), \
             patch.object(sys, 'argv', args), \
             patch.dict(os.environ, SLURM_ARRAY_JOB_ID='1', SLURM_ARRAY_TASK_ID='1'), \
             patch.object(os, 'execvp') as launch:
            sweep_launcher.main()
            return launch.call_args.args[1]

    def test_fresh_existing_and_explicit(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertFalse(any(x.startswith('--cg_resume_state=') for x in self.launch(root)))
            run = Path(root) / '1_0'
            run.mkdir()
            with self.assertRaisesRegex(ValueError, 'No valid'):
                self.launch(root)
            save(run, -1, 3)
            (run / 'wandb_id.txt').write_text('original')
            command = self.launch(root)
            self.assertIn('--cg_resume_state=' + str(run / 'cg_state_0'), command)
            self.assertIn('--wandb_run_id=original', command)
            other = Path(root) / 'other'
            save(other, -1, 1)
            explicit = str(other / 'cg_state_0')
            command = self.launch(root, ['--cg_resume_state=' + explicit,
                                         '--wandb_run_id=explicit'])
            self.assertIn('--cg_resume_state=' + explicit, command)
            self.assertIn('--wandb_run_id=explicit', command)
            with self.assertRaises(FileNotFoundError):
                self.launch(root, ['--cg_resume_state=' + str(other / 'cg_state_1')])

    def test_requested_missing(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, 'does not exist'):
                self.launch(root, ['--resume_from=99'])

    def test_swept_checkpoint_precedence(self):
        with tempfile.TemporaryDirectory() as root:
            other = Path(root) / 'other'
            save(other, -1, 1)
            checkpoint = str(other / 'cg_state_0')
            command = self.launch(root, ['--cg_resume_state=missing',
                                         '--cg_resume_state=' + checkpoint + '&invalid'])
            self.assertEqual([x for x in command if x.startswith('--cg_resume_state=')],
                             ['--cg_resume_state=' + checkpoint])
