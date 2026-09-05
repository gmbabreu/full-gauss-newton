# Decisions log

## 2026-09-05 — controlled outer decay for ordinary Muon--GN

* `outer_weight_decay` is a dimensionless shrink applied once to actual outer
  hidden rank-2 Muon matrices. Embedding, head, vectors, and scalars are excluded.
* Every searched and committed candidate uses `theta + alpha*d - rho*M*theta`.
  The raw inner iterate and optimizer state remain undecayed.
* Historical `optimizer_wd` behavior is unchanged: it reaches auxiliary AdamW
  leaves / Muon's `adam_weight_decay`, while Muon's matrix `weight_decay` remains
  omitted. The solver metric separation therefore isolates only the new decay.
* Historical `inner_loop_wd` is not repaired. In ordinary GN it changes the
  reported displacement penalty but is constant with respect to logits and hence
  does not change the applied GN gradient. `parameter_wd` remains unused.
* Historical `param_norm` remains the inner candidate norm, and historical
  `learning_rate` remains a post-update schedule lookup. New `outer/*` metrics use
  accepted outer weights and record pre-update inner schedule counters/LRs.
* `total_tokens` remains an inner-work convention and omits line-search data.
  The existing `single_batch_inner=True` double fetch is intentionally unchanged.
* Dataset state is not stored by the production checkpoint call and training RNG
  restarts. Full optimizer/warm-start restoration is honest, but old data-trajectory
  continuation is not claimed.
* An automatic relative-update controller is deferred until a reference trajectory
  identifies the layer/group ratio and its interaction with line search.
* The controlled A/B comparison explicitly disables weight averaging in both arms,
  avoiding different EMA frequencies in the historical no-search code paths.
