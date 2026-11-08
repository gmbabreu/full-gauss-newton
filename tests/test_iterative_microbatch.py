"""CPU ground-truth check of the production accumulation helper."""
import ast
from pathlib import Path
import unittest
import jax
import jax.numpy as jnp
import numpy as np
import optax

source = Path('EasyLM/models/llama/llama_train_gn.py')
tree = ast.parse(source.read_text())
names = {'microbatch_groups', 'weighted_microbatch_sum', 'iterative_gn_microbatch'}
ns = dict(jax=jax, jnp=jnp, linearize=jax.linearize,
          linear_transpose=jax.linear_transpose)
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                             and n.name in names], type_ignores=[]), str(source), 'exec'), ns)

class MicrobatchTest(unittest.TestCase):
    def test_explicit_operator_and_adam(self):
        p = jnp.linspace(-.3,.4,12)
        batch = dict(input_tokens=jnp.arange(42,dtype=jnp.float32).reshape(7,2,3)/30,
                     targets=jnp.arange(14).reshape(7,2)%3,
                     mask=jnp.array([[1,1],[1,0],[0,0],[1,1],[0,1],[1,1],[1,0]]))
        def logits(p,b):
            return jnp.tanh(b['input_tokens']@p[:9].reshape(3,3)+p[9:])
        def loss(z,b):
            lp=jnp.take_along_axis(jax.nn.log_softmax(z),b['targets'][...,None],-1)[...,0]
            return -jnp.mean(jnp.sum(jnp.where(b['mask']>0,lp,0),-1)/
                             jnp.maximum(jnp.sum(b['mask'],-1),1e-10))
        z=logits(p,batch)
        J=jax.jacfwd(lambda p:logits(p,batch).reshape(-1))(p)
        L=lambda z:loss(z.reshape(7,2,3),batch)
        g=J.T@jax.grad(L)(z.reshape(-1))
        G=J.T@jax.hessian(L)(z.reshape(-1))@J
        for n in (1,3,20):
            _,groups=ns['microbatch_groups'](7,n,1)
            def calc(q,last=True):
                return ns['iterative_gn_microbatch'](batch,groups,p,q,logits,loss,jnp.bool_(last))
            q=p+.07
            value,r,b=calc(q)
            np.testing.assert_allclose(r,g+G@(q-p),atol=1e-6)
            np.testing.assert_allclose(b,g,atol=1e-6)
            np.testing.assert_allclose(value,L(z.reshape(-1))+g@(q-p)+.5*(q-p)@G@(q-p),atol=1e-6)
            np.testing.assert_array_equal(calc(q,False)[2],jnp.zeros_like(p))
            tx=optax.chain(optax.clip_by_global_norm(.02),optax.adamw(.003,weight_decay=.01))
            a=b=q; sa=tx.init(q); sb=tx.init(q)
            for _ in range(20):
                ua,sa=tx.update(calc(a)[1],sa,a)
                ub,sb=tx.update(g+G@(b-p),sb,b)
                a=optax.apply_updates(a,ua); b=optax.apply_updates(b,ub)
            np.testing.assert_allclose(a,b,atol=1e-6)
            for x,y in zip(jax.tree.leaves(sa),jax.tree.leaves(sb)):
                np.testing.assert_allclose(x,y,atol=1e-6)

if __name__ == '__main__':
    unittest.main()
