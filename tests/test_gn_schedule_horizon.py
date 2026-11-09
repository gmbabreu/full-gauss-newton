"""CPU check of actual trainer schedule construction, without TPU imports."""
import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import optax

class HorizonTests(unittest.TestCase):
    def test_schedule_and_stopping_are_independent(self):
        tree=ast.parse(Path('EasyLM/models/llama/llama_train_gn.py').read_text())
        fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='get_global_lr_sched')
        assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
            and any(isinstance(t,ast.Name) and t.id=='lr_sched' for t in n.targets)
            and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name)
            and n.value.func.id=='get_global_lr_sched')
        code=compile(ast.Module(body=[fn,assignment],type_ignores=[]),'schedule','exec')
        for batch in (64,256,720,1024,2048):
            stop=299000000//(batch*1024)
            horizon=math.ceil(1000000000/(batch*1024))
            def schedule(total, explicit):
                flags=SimpleNamespace(lr_sched='global_cosine',inner_loop_lr=.001,
                    total_steps=total,schedule_steps=explicit,inner_loop_iter=50,
                    global_warmup=0,inner_loop_warmup=0,end_lr=0)
                ns={'FLAGS':flags,'optax':optax,'schedule_steps':explicit or total}; exec(code,ns);return ns['lr_sched']
            short=schedule(stop,horizon); long=schedule(horizon,0)
            for step in (0,1,stop//2,stop):
                observed=float(short(step*50))
                self.assertEqual(observed,float(long(step*50)))
                expected=.0005*(1+math.cos(math.pi*step/horizon))
                self.assertLess(abs(observed-expected),2e-10)
            self.assertAlmostEqual(float(schedule(stop,0)(stop*50)),0.)

if __name__ == '__main__': unittest.main()
