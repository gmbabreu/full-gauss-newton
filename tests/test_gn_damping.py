"""CPU checks of the trainer's actual nested GN gradient, without TPU imports."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import optax


class GNDampingTest(unittest.TestCase):
    def test_explicit_system_and_adam(self):
        source = Path(__file__).resolve().parents[1] / 'EasyLM/models/llama/llama_train_gn.py'
        module = ast.parse(source.read_text())
        main = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        step = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == 'train_step_gauss_newton')
        kernel = next(n for n in step.body if isinstance(n, ast.FunctionDef) and n.name == 'value_and_gradient')
        dot = lambda a, b: sum(jnp.vdot(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))
        flags = SimpleNamespace(gn_damping_mu=0.)
        # Unequal leaves catch accidental per-leaf mean penalties.
        p0 = {'w': jnp.array([.2, -.4]), 'b': jnp.array([.3])}
        def logits(p):
            x, y = p['w']; z = p['b'][0]
            return jnp.stack([jnp.sin(x)+y*y, x*z, jnp.exp(z)+y])
        loss = lambda z: -jax.nn.log_softmax(z)[1]
        env = dict(jax=jax, jnp=jnp, FLAGS=flags, f_batch=logits,
                   scalar_loss_on_logits=loss, linearize=jax.linearize,
                   linear_transpose=jax.linear_transpose, tree_dot=dot,
                   global_norm=lambda t: jnp.sqrt(dot(t, t)))
        exec(compile(ast.Module(body=[kernel], type_ignores=[]), str(source), 'exec'), env)
        from jax.flatten_util import ravel_pytree
        flat0, unravel = ravel_pytree(p0)
        f = lambda x: logits(unravel(x))
        z = f(flat0)
        J = jax.jacfwd(f)(flat0)
        G = J.T @ jax.hessian(loss)(z) @ J
        g = J.T @ jax.grad(loss)(z)
        d = jnp.array([.1, .3, -.2])
        for mu in (0., .03, .3):
            flags.gn_damping_mu = mu
            (value, _), (grad, bnorm) = env['value_and_gradient'](p0, unravel(flat0+d), jnp.bool_(True))
            expected = g + (G + mu*jnp.eye(3)) @ d
            error = float(jnp.max(jnp.abs(ravel_pytree(grad)[0]-expected)))
            print(f'mu={mu}: gradient max error={error:.3e}')
            np.testing.assert_allclose(ravel_pytree(grad)[0], expected, atol=2e-7)
            np.testing.assert_allclose(value, loss(z)+g@d+.5*d@G@d+.5*mu*(d@d), atol=2e-7)
            np.testing.assert_allclose(bnorm, jnp.linalg.norm(g), atol=2e-7)
        flags.gn_damping_mu = .3
        tx = optax.adam(.01)
        @jax.jit
        def solve():
            def body(_, carry):
                p, state = carry
                _, (grad, _) = env['value_and_gradient'](p0, p, jnp.bool_(True))
                updates, state = tx.update(grad, state, p)
                return optax.apply_updates(p, updates), state
            return jax.lax.fori_loop(0, 2000, body, (p0, tx.init(p0)))[0]
        actual = ravel_pytree(solve())[0] - flat0
        expected = jnp.linalg.solve(G+.3*jnp.eye(3), -g)
        residual = float(jnp.linalg.norm(g+(G+.3*jnp.eye(3))@actual)/jnp.linalg.norm(g))
        print(f'Adam vs dense solve: max error={float(jnp.max(jnp.abs(actual-expected))):.3e}; relative residual={residual:.3e}')
        np.testing.assert_allclose(actual, expected, atol=2e-6)


if __name__ == '__main__':
    unittest.main()
