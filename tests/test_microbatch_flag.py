import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

class FlagTest(unittest.TestCase):
    def test_alias_resolution(self):
        tree=ast.parse(Path('EasyLM/models/llama/llama_train_gn.py').read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        code=compile(ast.Module(body=main.body[:3],type_ignores=[]),'flags','exec')
        for new,old,want in [(0,1,1),(0,4,4),(4,1,4),(8,4,8)]:
            flags=SimpleNamespace(n_micro=new,cg_n_micro=old)
            exec(code,dict(FLAGS=flags))
            self.assertEqual((flags.n_micro,flags.cg_n_micro),(want,want))
        for new,old in [(-1,1),(0,0)]:
            with self.assertRaises(ValueError):
                exec(code,dict(FLAGS=SimpleNamespace(n_micro=new,cg_n_micro=old)))

    def test_resume_alias(self):
        tree=ast.parse(Path('EasyLM/cg_resume.py').read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_flags')
        ns={}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'resume','exec'),ns)
        check=ns['validate_flags']
        check({'cg_n_micro':4},{'n_micro':4,'cg_n_micro':4})
        check({'cg_n_micro':1},{'n_micro':0,'cg_n_micro':1})
        with self.assertRaises(ValueError):
            check({'cg_n_micro':4},{'n_micro':8,'cg_n_micro':8})

if __name__=='__main__': unittest.main()
