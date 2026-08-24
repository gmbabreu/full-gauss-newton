"""Independent validation for the matrix-free CG matrix-norm diagnostics."""

import itertools

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def tree_dot(left, right):
    return sum(jnp.vdot(x, y) for x, y in zip(jax.tree.leaves(left), jax.tree.leaves(right)))


def main():
    theta = jnp.array([0.3, -0.7, 0.2], dtype=jnp.float64)
    targets = jnp.array([0.2, 0.5, 0.3], dtype=jnp.float64)

    def model(p):
        return jnp.array([p[0] * p[1] + jnp.sin(p[2]), p[0] ** 2 - p[1], p[2] * p[1]])

    def loss_on_logits(logits):
        return -jnp.sum(targets * jax.nn.log_softmax(logits))

    logits = model(theta)
    jacobian = jax.jacobian(model)(theta)
    logit_hessian = jax.hessian(loss_on_logits)(logits)
    g_dense = jacobian.T @ logit_hessian @ jacobian

    logits0, jvp_fn = jax.linearize(model, theta)
    loss_grad = jax.grad(loss_on_logits)
    jt_fn = jax.linear_transpose(jvp_fn, theta)

    call_count = 0

    def apply_g(v):
        nonlocal call_count
        call_count += 1
        jv = jvp_fn(v)
        _, hjv = jax.jvp(loss_grad, (logits0,), (jv,))
        return jt_fn(hjv)[0]

    v = jnp.array([0.4, -0.1, 0.8], dtype=jnp.float64)
    g_v = apply_g(v)
    dense_g_v = g_dense @ v
    gv_abs = jnp.linalg.norm(g_v - dense_g_v)
    gv_rel = gv_abs / (jnp.linalg.norm(dense_g_v) + 1e-12)

    interpolation_lambda = 0.37
    eta = 0.01
    adam_diag = jnp.array([0.11, 0.23, 0.07], dtype=jnp.float64)
    old_av = interpolation_lambda * dense_g_v + (1 - interpolation_lambda) / eta * adam_diag * v
    new_av = interpolation_lambda * apply_g(v) + (1 - interpolation_lambda) / eta * adam_diag * v
    av_abs = jnp.linalg.norm(old_av - new_av)
    av_rel = av_abs / (jnp.linalg.norm(old_av) + 1e-12)

    signs = [jnp.asarray(z, dtype=jnp.float64) for z in itertools.product((-1.0, 1.0), repeat=theta.size)]
    frob_sum = 0.0
    for z in signs:
        gz = apply_g(z)
        frob_sum += tree_dot((gz,), (gz,))
    g_frob = jnp.sqrt(frob_sum / len(signs))
    dense_frob = jnp.linalg.norm(g_dense, ord="fro")

    probes, power_iters = len(signs), 40
    q = jnp.array([1.0, -1.0, 1.0], dtype=jnp.float64)
    q /= jnp.linalg.norm(q)
    power_calls_before = call_count
    for _ in range(power_iters - 1):
        gq = apply_g(q)
        q = gq / (jnp.linalg.norm(gq) + 1e-12)
    gq = apply_g(q)  # final call is reused below
    power_calls = call_count - power_calls_before
    g_spectral = tree_dot((q,), (gq,)) / tree_dot((q,), (q,))
    g_residual = jnp.linalg.norm(gq - g_spectral * q) / (jnp.linalg.norm(gq) + 1e-12)
    dense_spectral = jnp.linalg.eigvalsh(g_dense)[-1]

    d_diag = adam_diag / eta
    d_dense = jnp.diag(d_diag)
    d_frob = jnp.sqrt(jnp.sum(d_diag**2))
    d_spectral = jnp.max(d_diag)
    d_min = jnp.min(d_diag)
    d_max = jnp.max(d_diag)
    d_condition = jnp.where(d_min > 0, d_max / d_min, jnp.inf)
    dense_d_values = (jnp.linalg.norm(d_dense, "fro"), jnp.linalg.norm(d_dense, 2), jnp.linalg.eigvalsh(d_dense)[0], jnp.linalg.eigvalsh(d_dense)[-1], jnp.linalg.cond(d_dense))
    for actual, expected in zip((d_frob, d_spectral, d_min, d_max, d_condition), dense_d_values):
        assert jnp.allclose(actual, expected)

    ratio_frob = g_frob / (d_frob + 1e-12)
    ratio_spec = g_spectral / (d_spectral + 1e-12)
    dense_ratio_frob = dense_frob / jnp.linalg.norm(d_dense, "fro")
    dense_ratio_spec = dense_spectral / jnp.linalg.norm(d_dense, 2)
    lambda_frob = d_frob / (g_frob + d_frob + 1e-12)
    lambda_spec = d_spectral / (g_spectral + d_spectral + 1e-12)
    extra_calls = probes + power_iters

    assert jnp.allclose(g_v, dense_g_v, rtol=1e-11, atol=1e-12)
    assert jnp.allclose(old_av, new_av, rtol=1e-11, atol=1e-12)
    assert jnp.allclose(g_frob, dense_frob, rtol=1e-11, atol=1e-12)
    assert jnp.allclose(d_spectral, d_max)
    assert jnp.allclose(ratio_frob, dense_ratio_frob, rtol=1e-11)
    assert jnp.allclose(ratio_spec, dense_ratio_spec, rtol=1e-7)
    assert jnp.allclose(lambda_frob * g_frob, (1 - lambda_frob) * d_frob, rtol=1e-10)
    assert jnp.allclose(lambda_spec * g_spectral, (1 - lambda_spec) * d_spectral, rtol=1e-10)
    assert power_calls == power_iters
    assert extra_calls == probes + power_iters

    rank_deficient_g = jnp.array([[2.0, 2.0], [2.0, 2.0]])
    print(f"Matrix-free Gv absolute error: {float(gv_abs):.12e}")
    print(f"Matrix-free Gv relative error: {float(gv_rel):.12e}")
    print(f"Old Av vs refactored Av absolute error: {float(av_abs):.12e}")
    print(f"Old Av vs refactored Av relative error: {float(av_rel):.12e}")
    print(f"G_frob exact dense: {float(dense_frob):.12e}")
    print(f"G_frob exhaustive estimator: {float(g_frob):.12e}")
    print(f"G_spectral exact dense: {float(dense_spectral):.12e}")
    print(f"G_spectral power estimate: {float(g_spectral):.12e}")
    print(f"G_spectral relative error: {float(abs(g_spectral-dense_spectral)/dense_spectral):.12e}")
    print(f"G_spectral_relative_residual: {float(g_residual):.12e}")
    print(f"D_frob: {float(d_frob):.12e} (dense {float(dense_d_values[0]):.12e})")
    print(f"D_spectral: {float(d_spectral):.12e} (dense {float(dense_d_values[1]):.12e})")
    print(f"D_min_eig: {float(d_min):.12e} (dense {float(dense_d_values[2]):.12e})")
    print(f"D_max_eig: {float(d_max):.12e} (dense {float(dense_d_values[3]):.12e})")
    print(f"D_condition: {float(d_condition):.12e} (dense {float(dense_d_values[4]):.12e})")
    print(f"D_spectral == D_max_eig: {bool(d_spectral == d_max)}")
    print(f"G_D_ratio_frob: {float(ratio_frob):.12e} (dense {float(dense_ratio_frob):.12e})")
    print(f"G_D_ratio_spec: {float(ratio_spec):.12e} (dense {float(dense_ratio_spec):.12e})")
    print(f"Frobenius balance sides: {float(lambda_frob*g_frob):.12e}, {float((1-lambda_frob)*d_frob):.12e}")
    print(f"Spectral balance sides: {float(lambda_spec*g_spectral):.12e}, {float((1-lambda_spec)*d_spectral):.12e}")
    print(f"Extra operator calls: {extra_calls} = {probes} probes + {power_iters} power iterations")
    print(f"Power-iteration calls (final Gq reused): {power_calls}")
    print(f"Rank-deficient G eigenvalues: {jnp.linalg.eigvalsh(rank_deficient_g)}")
    print("Raw G_condition intentionally omitted because the zero eigenvalue makes it infinite.")


if __name__ == "__main__":
    main()
