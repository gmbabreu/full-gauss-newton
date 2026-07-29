"""
gn_train_step.py -- Unified Gauss-Newton inner-solver module.

================
Before this refactor, the Muon and CG inner-loop solvers each had their own
independent copy of: f_batch, scalar_loss_on_logits, linearize/jvp/
linear_transpose setup, and the g0/Hv/grad_params combination logic. That
duplication was the main source complexity we set out to remove.

Both Muon and CG are solving the same local quadratic model of the loss,
    Q(offset) = L(params0) + b^T offset + 1/2 offset^T G offset
where G = J0^T H J0 is the Gauss-Newton matrix and b = J0^T g0. They just
solve it differently. Muon uses iterative first-order steps on Q, whereas CG 
 uses krylov to solve G @ offset = -b). This module makes that quadratic model
build_gn_operators(...) into ONE object (GNOperators), and both solvers
consume it identically, neither one redefines the model or the loss. 
This should also make it simpler to include other optimizers in the future

ARCHITECTURE
============
    train_step_gn
        |
        +-- build_gn_operators(params0, batch, ...)  -> GNOperators
        |       (the only place f_batch / scalar_loss_on_logits /
        |        linearize / linear_transpose are defined)
        |
        +-- solve_inner_problem(gn_ops, solver_type, ...) -> offset, metrics
        |       +-- solve_inner_muon (runs the full inner_loop_iter loop
        |       |    internally via jax.lax.scan)
        |       +-- solve_inner_cg   (one full jax.scipy.sparse.linalg.cg
        |            solve to cg_maxiter, fresh each call)
        |
        +-- (linesearch stays OUTSIDE this module, in the outer training
        |    loop, intentionally evaluates on a separate, freshly
        |    sampled batch)
        |
        +-- returns candidate_params for the caller to linesearch/apply
"""
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import optax
from jax import linearize, linear_transpose


def global_norm(tree):
    """ L2 norm of an entire pytree, treated as one flat vector. """
    squared = jax.tree_util.tree_map(lambda x: jnp.sum(jnp.square(x)), tree)
    flattened, _ = jax.flatten_util.ravel_pytree(squared)
    return jnp.sqrt(jnp.sum(flattened))


def tree_dot(a, b):
    """ Dot product <a, b> of two pytrees with identical structure, treated
    as one flat vector each. """
    products = jax.tree_util.tree_map(lambda x, y: jnp.sum(x * y), a, b)
    flattened, _ = jax.flatten_util.ravel_pytree(products)
    return jnp.sum(flattened)


# ============================================================
# The shared mathematical object for any solver
# previously, different solver's inner-loop
# code each independently redefined f_batch/scalar_loss_on_logits/
# linearize/linear_transpose. Now there's exactly one place that happens.
# ============================================================
class GNOperators(NamedTuple):
    b_params: Any            # J0^T g0_logits -- the linear ("b") term of the local quadratic model
    apply: Callable          # v -> J0^T H J0 v  (the Gauss-Newton matrix-vector product; param-space in/out)
    quadratic_loss: Callable  # offset -> local quadratic model value Q(params0 + offset)
    base_loss: jnp.ndarray   # scalar_loss_on_logits(logits0) -- the constant term of Q


def build_gn_operators(params0, rng, batch, wd, model, LLaMAConfigurator,
                        cross_entropy_loss_and_accuracy_with_weight_decay,
                        with_sharding_constraint, PS, JaxRNG):
    """
    Builds the local Gauss-Newton quadratic model at fixed linearization
    point params0, for the given batch. This is the place where f_batch /
    scalar_loss_on_logits / linearize / linear_transpose get defined.
    both solve_inner_muon and solve_inner_cg consume the resulting
    GNOperators; neither redefines any of this itself.

    NOTE: this builds ONE fixed linearization for the given batch.
    Callers that need a fresh linearization per inner-loop iteration (the
    original code's single_batch_inner=False behavior) must call this
    function again per iteration themselves -- see the module docstring's
    "Scope / known limitation" section.
    """
    rng_generator = JaxRNG(rng)
    batch_ = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
    fixed_rngs = rng_generator(LLaMAConfigurator.rng_keys())

    def f_batch(p):
        out = model.apply(
            p, batch_['input_tokens'], deterministic=False,
            rngs=fixed_rngs,
        )
        return out.logits

    def scalar_loss_on_logits(logits):
        loss, _ = cross_entropy_loss_and_accuracy_with_weight_decay(
            logits, batch_['target_tokens'], params0, params0, batch_['loss_masks'], weight_decay=wd
        )
        return loss

    logits0, jvp_fn = linearize(f_batch, params0)
    grad_Ly = jax.grad(scalar_loss_on_logits)
    g0_logits = grad_Ly(logits0)
    jt_fn = linear_transpose(jvp_fn, params0)
    (b_params,) = jt_fn(g0_logits)
    base_loss = scalar_loss_on_logits(logits0)

    def apply_G(v):
        """ v -> J0^T H J0 v, the Gauss-Newton matrix-vector product.
        Parameter-space in, parameter-space out. """
        logits_v = jvp_fn(v)
        _, Hv_logits = jax.jvp(grad_Ly, (logits0,), (logits_v,))
        (Gv_params,) = jt_fn(Hv_logits)
        return Gv_params

    def evaluate_quadratic_model(offset):
        """
        Q(offset) = base_loss + b.offset + 1/2 offset^T G offset, plus a
        weight-decay term
        """
        quadratic_part = base_loss + tree_dot(b_params, offset) + 0.5 * tree_dot(offset, apply_G(offset))
        l2_total = wd * sum(jnp.mean(o ** 2) for o in jax.tree_util.tree_leaves(offset))
        return quadratic_part + l2_total

    gn_ops = GNOperators(
        b_params=b_params, apply=apply_G,
        quadratic_loss=evaluate_quadratic_model, base_loss=base_loss,
    )
    return gn_ops, rng_generator


# ============================================================
# Muon inner solver -- runs its FULL inner loop internally via
# jax.lax.scan. take a GNOperators and params0, and return a finished offset
# ============================================================
def solve_inner_muon(gn_ops, params0, offset0, opt_state0, tayl_solver, inner_loop_iter):
    """
    Runs `inner_loop_iter` Muon (or whatever optax transform tayl_solver
    is) steps on the fixed quadratic model gn_ops, starting from offset0.
    """
    def body(carry, _):
        offset, opt_state = carry

        #   The old implementation computed:
        #
        #       J^T(g0 + H @ v)
        #
        #   The refactored implementation computes:
        #
        #       J^T(g0) + J^T(H @ v)
        #
        #   These are mathematically identical by linearity of the
        #   transpose Jacobian. They are not necessarily bitwise
        #   identical in fp32 because of floating-point addition
        grad_params = jax.tree_util.tree_map(lambda b, g: b + g, gn_ops.b_params, gn_ops.apply(offset))

        # Decoupled weight decay (optax.adamw's weight_decay /
        # optax.contrib.muon's adam_weight_decay) reads its decay term
        # from whatever `params` argument is passed to .update() here.
        absolute_params = jax.tree_util.tree_map(lambda p0, o: p0 + o, params0, offset)
        updates, opt_state = tayl_solver.update(grad_params, opt_state, absolute_params)

        offset = optax.apply_updates(offset, updates)
        return (offset, opt_state), grad_params

    (offset, opt_state), grad_params_history = jax.lax.scan(
        body, (offset0, opt_state0), xs=None, length=inner_loop_iter
    )
    # logging
    final_grad_params = jax.tree_util.tree_map(lambda x: x[-1], grad_params_history)
    metrics = {
        'gradient_norm': global_norm(final_grad_params),
        'b_norm': global_norm(gn_ops.b_params),
        'relative_residual': global_norm(final_grad_params) / (global_norm(gn_ops.b_params) + 1e-12),
    }
    return offset, opt_state, metrics


# ============================================================
# CG inner solver: one full solve to cg_maxiter, optionally warm-started.
# ============================================================
def solve_inner_cg(gn_ops, params0, cg_tol, cg_atol, cg_maxiter, cg_damping, x0=None):
    """
    Solves (G + cg_damping*I) @ offset = -b_params via jax.scipy.sparse.linalg.cg.

    If x0 is provided, CG warm-starts from that previous offset; otherwise it
    preserves the historical behavior and starts from a zero offset.

    cg_damping exists because plain (undamped) CG was empirically found to
    diverge at higher cg_maxiter on the real model.
    """
    rhs = jax.tree_util.tree_map(lambda x: -x, gn_ops.b_params)

    def Gv_damped(v):
        Gv = gn_ops.apply(v)
        return jax.tree_util.tree_map(lambda g, vi: g + cg_damping * vi, Gv, v)

    if x0 is None:
        x0 = jax.tree_util.tree_map(jnp.zeros_like, params0)

    offset, _ = jax.scipy.sparse.linalg.cg(
        Gv_damped, rhs,
        x0=x0,
        tol=cg_tol, atol=cg_atol, maxiter=cg_maxiter,
    )
    # Residual r = G.offset + b (should be ~0 if CG converged); used for
    # the relative_residual diagnostic, the standard CG convergence metric.
    Gx = gn_ops.apply(offset)
    residual = jax.tree_util.tree_map(lambda gx, b: gx + b, Gx, gn_ops.b_params)
    residual_norm = global_norm(residual)
    b_norm = global_norm(gn_ops.b_params)
    metrics = {
        'gradient_norm': residual_norm,
        'b_norm': b_norm,
        'relative_residual': residual_norm / (b_norm + 1e-12),
    }
    return offset, metrics


# ============================================================
# solving the problem
# ============================================================
def solve_inner_problem(gn_ops, solver_type, params0, **solver_kwargs):
    if solver_type == 'muon':
        return solve_inner_muon(
            gn_ops, params0, solver_kwargs['offset0'], solver_kwargs['opt_state0'],
            solver_kwargs['tayl_solver'], solver_kwargs['inner_loop_iter'],
        )
    elif solver_type == 'cg':
        offset, metrics = solve_inner_cg(
            gn_ops, params0, solver_kwargs['cg_tol'], solver_kwargs['cg_atol'],
            solver_kwargs['cg_maxiter'], solver_kwargs['cg_damping'],
            solver_kwargs.get('x0', None),
        )
        return offset, None, metrics  # CG has no optimizer state to return
    else:
        raise ValueError(f"Unknown solver_type: {solver_type!r} (expected 'muon' or 'cg')")


# ============================================================
# The single unified train step. 
# ============================================================
def train_step_gn(params0, rng, batch, wd, solver_type, model, LLaMAConfigurator,
                   cross_entropy_loss_and_accuracy_with_weight_decay,
                   with_sharding_constraint, PS, JaxRNG, **solver_kwargs):
    """
    solver_type selects 'muon' or 'cg'; solver_kwargs are passed straight
    through to whichever solver is selected (offset0/opt_state0/
    inner_loop_iter/tayl_solver for muon; cg_tol/cg_atol/cg_maxiter/
    cg_damping for cg).
    """
    gn_ops, rng_generator = build_gn_operators(
        params0, rng, batch, wd, model, LLaMAConfigurator,
        cross_entropy_loss_and_accuracy_with_weight_decay,
        with_sharding_constraint, PS, JaxRNG,
    )
    offset, opt_state, solver_metrics = solve_inner_problem(gn_ops, solver_type, params0, **solver_kwargs)
    candidate_params = jax.tree_util.tree_map(lambda p0, o: p0 + o, params0, offset)

    metrics = dict(solver_metrics)
    metrics['linear_model_loss'] = gn_ops.quadratic_loss(offset)
    metrics['param_norm'] = global_norm(candidate_params)

    return candidate_params, opt_state, rng_generator(), metrics
