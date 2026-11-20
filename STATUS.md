# Status

The controlled GN prefix diagnostic is implemented on
`codex/gn-prefix-transfer-diagnostic-main`, created directly from freshly fetched
`origin/main` (ede9de0). Only diagnostic implementation, tests, an independent
launch preset and relevant documentation are ported. No queue infrastructure or
GN sweep commits are included. Earlier work is preserved on
`gn-prefix-transfer-diagnostic`.

The committed trajectory remains K10. Additional K20/K50/K100 prefixes are
observational. The experiment plan probes early/middle/later outer steps, using
one fixed solve batch per probe, the ordinary LS data/RNG and an independent
held-out loader. Compare accepted and matched-norm held-out improvements against
K10 while preserving the K10 optimizer/LR sequence and training configuration.
See [docs/gn_prefix_probe.md](docs/gn_prefix_probe.md).

CPU validation passed against the untouched main baseline: 29 tests and 60
subtests, including serialized state/RNG/log/data equality across two outer updates
with both AdamW and Muon. All 42 launcher tests passed. The diagnostic has a
standalone preset using the existing main launcher; production launcher code is
unchanged. No TPU training,
deployment or PR merge is authorized or performed. Live multi-host memory and
scientific validation remain pending. Spectral diagnostics and adaptive
scheduling are out of scope.
