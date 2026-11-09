import ast
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from gn_sweep_queue import build_plan, verdict, TRIALS

class SweepTests(unittest.TestCase):
    def test_budgets_and_resolved_config(self):
        for trial, batch, lr, micro in TRIALS:
            p = build_plan(trial, 299000000, 1000000000)
            f = p['flags']; unit = batch*1024
            n = int(f['total_steps']); h = int(f['schedule_steps'])
            self.assertLessEqual(n*unit, 299000000)
            self.assertGreater((n+1)*unit, 299000000)
            self.assertGreaterEqual(h*unit, 1000000000)
            self.assertLess((h-1)*unit, 1000000000)
            self.assertEqual(f['dtype'], 'fp32')
            self.assertEqual(f['eval_steps'], '8')
            self.assertEqual(f['eval_freq'], f['log_freq'])
            self.assertEqual(f['save_model_freq'], '0')
            self.assertEqual(f['single_batch_inner'], 'True')
            self.assertEqual(f['train_dataset.huggingface_dataset.batch_size'], str(batch))
            self.assertEqual(f['eval_dataset.huggingface_dataset.batch_size'], 128)
            self.assertEqual(f['n_micro'], str(micro))
            self.assertEqual(float(f['inner_loop_lr']), lr)
            self.assertEqual(f['load_dataset_state'], '')
            self.assertEqual(p['experiment_id'], p['run_id']+'_0')

    def test_bad_budget(self):
        for tokens, horizon in [(1, 1000000000), (299000000, 1000000)]:
            with self.assertRaises(ValueError): build_plan(0, tokens, horizon)

    def test_worker_completion(self):
        good = dict(run_id='abc', state='exited', exit_status=0)
        self.assertTrue(verdict({0:good, 1:good}, 'abc'))
        for state in ('started','prepared'):
            self.assertFalse(verdict({0:good,1:dict(good,state=state,exit_status=None)}, 'abc'))
        for bad in [dict(good,state='cancelled'), dict(good,exit_status=1),
                    dict(good,run_id='other'), dict(good,state='unknown')]:
            with self.assertRaises(RuntimeError): verdict({0:good,1:bad},'abc')
        with self.assertRaises(RuntimeError): verdict({},'abc')

if __name__ == '__main__': unittest.main()
