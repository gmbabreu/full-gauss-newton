# Validated implementation findings

- 2026-11-20: The production non-CG GN inner-step pjit on main has argument
  donation disabled. Sharing immutable K10 buffers is safe under this existing
  contract; continuation replaces only a rolling local state.
- 2026-11-20: CPU tests with real Optax AdamW and Muon confirm schedule counters
  must be read from ScaleByScheduleState before updating, rather than using the
  existing post-update learning_rate metric or assuming TrainState.step agrees.
- 2026-11-20: Dense Jacobian/logits-Hessian CPU checks validate raw/accepted GN
  predictions and post-update residuals through K100 on a tiny nonlinear model.
  These checks provide no evidence about held-out transfer during LLaMA training.
- 2026-11-20: Freshly fetched origin/main is ede9de0. Its GN trainer is byte-for-byte
  identical to the queue-source trainer before diagnostic changes, so the diagnostic
  ports without importing queue helpers or optimizer/sweep changes.
- 2026-11-20: Executing the untouched main host branch versus the enabled
  diagnostic with real AdamW and Muon produces byte-identical serialized committed
  and inner states, identical RNG, ordinary logs and LS pulls across two outer
  updates in the CPU harness. The main-based suite passed 29 tests and 60 subtests.
