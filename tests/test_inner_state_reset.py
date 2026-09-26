import unittest

import jax
import jax.numpy as jnp
import optax

from EasyLM.jax_utils import CustomTrainState, create_reset_train_state


class InnerStateResetTest(unittest.TestCase):
    def assert_trees_equal(self, left, right):
        self.assertEqual(jax.tree.structure(left), jax.tree.structure(right))
        for left_leaf, right_leaf in zip(
                jax.tree.leaves(left), jax.tree.leaves(right)):
            self.assertEqual(left_leaf.dtype, right_leaf.dtype)
            self.assertTrue(jnp.array_equal(left_leaf, right_leaf))

    def test_reset_preserves_state_fields_and_optimizer_semantics(self):
        tx = optax.adamw(learning_rate=0.01)
        initial_params = {'weight': jnp.array([1., -2.], dtype=jnp.float32)}
        state = CustomTrainState.create(
            apply_fn=None, params=initial_params, tx=tx)
        gradient = {'weight': jnp.array([0.3, -0.4], dtype=jnp.float32)}
        state = state.apply_gradients(grads=gradient).replace(
            step=jnp.array(7, dtype=jnp.asarray(state.step).dtype),
            warmstart_params={'weight': jnp.array([5., 6.], dtype=jnp.float32)},
        )
        outer_params = {'weight': jnp.array([4., -3.], dtype=jnp.float32)}

        old_reset = state.replace(
            params=outer_params, opt_state=tx.init(outer_params))
        new_reset = create_reset_train_state(
            CustomTrainState,
            step=state.step,
            apply_fn=state.apply_fn,
            params=outer_params,
            tx=state.tx,
            warmstart_params=state.warmstart_params,
        )

        self.assertIs(new_reset.apply_fn, state.apply_fn)
        self.assertIs(new_reset.tx, state.tx)
        self.assert_trees_equal(old_reset, new_reset)
        self.assert_trees_equal(old_reset.opt_state, tx.init(outer_params))

        next_gradient = {'weight': jnp.array([-0.2, 0.1], dtype=jnp.float32)}
        self.assert_trees_equal(
            old_reset.apply_gradients(grads=next_gradient),
            new_reset.apply_gradients(grads=next_gradient),
        )

    def test_reset_disabled_keeps_original_state(self):
        tx = optax.adamw(learning_rate=0.01)
        state = CustomTrainState.create(
            apply_fn=None,
            params={'weight': jnp.array([1.], dtype=jnp.float32)},
            tx=tx,
        )
        reset_start = False
        result = state
        if reset_start:
            result = create_reset_train_state(
                CustomTrainState, step=state.step, apply_fn=state.apply_fn,
                params=state.params, tx=state.tx,
                warmstart_params=state.warmstart_params)
        self.assertIs(result, state)


if __name__ == '__main__':
    unittest.main()
