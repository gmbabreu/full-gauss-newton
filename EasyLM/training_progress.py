"""Host-side schema-v2 training progress accounting.

These counters are reporting-only: they must never be used as optimizer or
learning-rate schedule state.
"""
from dataclasses import asdict, dataclass

PROGRESS_SCHEMA_VERSION = 2
TOKEN_CONVENTION = "distinct global input-token positions used by solve or line search"


@dataclass
class TrainingProgress:
    step_offset: int = 0
    token_offset: int = 0
    phase_completed_updates: int = 0
    phase_solve_tokens: int = 0
    phase_linesearch_tokens: int = 0
    phase_skipped_tokens: int = 0
    dataset_total_tokens: int = 0
    comparison_origin: str = "zero-origin experiment"

    def __post_init__(self):
        for name in asdict(self):
            if name != "comparison_origin" and getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")

    @property
    def completed_updates(self):
        return self.step_offset + self.phase_completed_updates

    @property
    def cumulative_tokens(self):
        return self.token_offset + self.phase_solve_tokens + self.phase_linesearch_tokens

    def charge(self, role, batch, dataset_metadata=None):
        tokens = int(batch["input_tokens"].size)
        field = {"solve": "phase_solve_tokens", "linesearch": "phase_linesearch_tokens",
                 "skipped": "phase_skipped_tokens"}.get(role)
        if field is None:
            raise ValueError(f"unknown training batch role: {role}")
        setattr(self, field, getattr(self, field) + tokens)
        if dataset_metadata and "dataset_total_tokens" in dataset_metadata:
            self.dataset_total_tokens = int(dataset_metadata["dataset_total_tokens"])
        else:
            self.dataset_total_tokens += tokens
        return tokens

    def complete_update(self):
        self.phase_completed_updates += 1

    def record(self, local_outer_step, **metrics):
        value = self.completed_updates
        total_tokens = self.cumulative_tokens
        return dict(metrics, progress_schema_version=PROGRESS_SCHEMA_VERSION,
                    completed_updates=value, step=value, global_step=value,
                    local_outer_step=int(local_outer_step),
                    phase_completed_updates=self.phase_completed_updates,
                    total_tokens=total_tokens, cumulative_tokens=total_tokens,
                    phase_solve_tokens=self.phase_solve_tokens,
                    phase_linesearch_tokens=self.phase_linesearch_tokens,
                    phase_skipped_tokens=self.phase_skipped_tokens,
                    dataset_total_tokens=self.dataset_total_tokens)

    def state_dict(self):
        return dict(asdict(self), progress_schema_version=PROGRESS_SCHEMA_VERSION,
                    token_convention=TOKEN_CONVENTION)

    @classmethod
    def from_state_dict(cls, state):
        if state.get("progress_schema_version") != PROGRESS_SCHEMA_VERSION:
            raise ValueError("unsupported training progress metadata")
        values = {name: state[name] for name in cls.__dataclass_fields__}
        return cls(**values)


def resolve_progress(step_override=-1, token_override=-1, saved_state=None, *, branch=False):
    """Resolve reporting offsets without affecting training state.

    Full resumes inherit phase counters. Parameter-only branches reset phase
    counters and use explicit offsets (or zero for a new experiment).
    """
    if saved_state is not None and not branch:
        saved = TrainingProgress.from_state_dict(saved_state)
        for name, override, expected in (("step", step_override, saved.step_offset),
                                         ("token", token_override, saved.token_offset)):
            if override >= 0 and override != expected:
                raise ValueError(f"log_{name}_offset conflicts with checkpoint reporting state")
        return saved
    if branch and saved_state is None and (step_override < 0) != (token_override < 0):
        raise ValueError("cumulative branches require both reporting offsets")
    step = max(step_override, 0)
    tokens = max(token_override, 0)
    origin = "explicit parent checkpoint" if step or tokens else "zero-origin experiment"
    return TrainingProgress(step_offset=step, token_offset=tokens,
                            dataset_total_tokens=tokens, comparison_origin=origin)


def configure_wandb_run(run):
    run.define_metric("*", step_metric="total_tokens")
    for name in ("total_tokens", "cumulative_tokens", "completed_updates", "step", "global_step"):
        run.define_metric(name, hidden=True)
