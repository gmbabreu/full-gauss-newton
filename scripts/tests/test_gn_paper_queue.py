import copy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tpu_resident_queue import make_queue_plan, parse_spec

class PaperQueueTests(unittest.TestCase):
    def test_resolved_protocol(self):
        spec = json.loads((Path(__file__).resolve().parents[1] / 'sweeps/gn_paper_300m.json').read_text())
        expected = [(2048,.001,.001),(4096,.002,.001),(3072,.002,.001),(2048,.002,.001),(1024,.002,.001),(4096,.004,.001),(2048,.001,0),(2048,.001,.01)]
        plan = make_queue_plan(spec)
        self.assertEqual(len(plan['trials']), 8)
        for t, (b,lr,wd) in zip(plan['trials'], expected):
            f=t['plan']['flags']
            self.assertEqual((int(f['train_dataset_batch_size']),float(f['inner_loop_lr']),float(f['optimizer_wd'])), (b,lr,wd))
            self.assertEqual(f['notes'],t['name'])
            self.assertEqual(int(f['schedule_steps']),2849988096//(b*1024))
            self.assertEqual(int(f['eval_steps']),50)
            self.assertEqual(int(f['train_dataset.huggingface_dataset.shuffle_seed']),42)
            self.assertTrue(f['log_initial_eval'])
            self.assertLessEqual(t['actual_solve_tokens'],300000000)
            self.assertLess(300000000-t['actual_solve_tokens'],b*1024)
            self.assertEqual(int(f['save_model_freq']),0)
        bad=copy.deepcopy(spec);bad['trials'][0]['optimizer_wd']=-1
        with self.assertRaises(ValueError): parse_spec(bad)

    def test_long_protocol(self):
        spec = json.loads((Path(__file__).resolve().parents[1] / 'sweeps/gn_paper_long.json').read_text())
        plan = make_queue_plan(spec)
        self.assertEqual(plan['trial_hours'],120)
        expected=[(.002,.001),(.002,.01),(.003,.001),(.001,.001)]
        self.assertEqual(len(plan['trials']),4)
        for t, pair in zip(plan['trials'],expected):
            f=t['plan']['flags']
            self.assertEqual((float(f['inner_loop_lr']),float(f['optimizer_wd'])),pair)
            self.assertEqual(int(f['total_steps']),1358)
            self.assertEqual(int(f['schedule_steps']),1358)
            self.assertEqual(int(f['eval_steps']),50)
            self.assertEqual(int(f['eval_freq']),24)
            self.assertEqual(int(f['n_micro']),4)
            self.assertEqual(int(f['ls_reuse_lag']),100)
            self.assertTrue(t['name'].endswith('_reuse100'))
            self.assertEqual(t['actual_solve_tokens'],2847932416)
            self.assertEqual(int(f['save_model_freq']),0)
        spec['trial_hours']=0
        with self.assertRaises(ValueError): parse_spec(spec)

    def test_reuse_is_optional_and_validated(self):
        spec = json.loads((Path(__file__).resolve().parents[1] / 'sweeps/gn_paper_300m.json').read_text())
        self.assertTrue(all('ls_reuse_lag' not in t['plan']['flags']
                            for t in make_queue_plan(spec)['trials']))
        for invalid in (-1, 1.5, True):
            spec['ls_reuse_lag'] = invalid
            with self.assertRaises(ValueError): parse_spec(spec)
