# Status

Optional raw CG matrix-norm logging is implemented. When
`cg_log_matrix_norms=True`, each outer CG step reports exactly:

- `G_frob`
- `G_spectral`
- `G_spectral_relative_residual`
- `D_frob`
- `D_spectral`
- `D_min_eig`
- `D_max_eig`
- `D_condition`
- `G_D_ratio_frob`
- `G_D_ratio_spec`
- `cg_lambda_balance_frob`
- `cg_lambda_balance_spec`
- `cg_matrix_norm_extra_operator_calls`

Diagnostic D is the complete Adam-side matrix and includes the `1 / eta`
factor. `G_condition` is intentionally omitted because raw Gauss–Newton may
be singular.

## Next action

Run the fixed-eta lambda experiment with the Adam learning-rate schedule held
constant at 0.01.
