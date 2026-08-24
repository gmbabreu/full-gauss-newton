# Decisions log

- 2026-08-24 (validated): Diagnostic `D_*` names denote the full Adam-side
  matrix, including `1 / eta`. Raw `G_condition` is not logged because
  Gauss–Newton may be singular, and a positive-spectrum condition number
  would require an explicit eigenvalue threshold.
