"""Host-side schema-v3 training progress accounting.

These counters are reporting-only: they must never be used as optimizer or
learning-rate schedule state.
"""
from dataclasses import asdict, dataclass
import time

PROGRESS_SCHEMA_VERSION = 3
TOKEN_CONVENTION = (
    "distinct global input-token positions used by solve; "
    "excludes line search and skipped fetches"
)


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
        return self.token_offset + self.phase_solve_tokens

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
        if state.get("progress_schema_version") not in (2, PROGRESS_SCHEMA_VERSION):
            raise ValueError("unsupported training progress metadata")
        # V2 already stored separate counters. Retain them and the explicit
        # parent prefix; only the derived token axis changes. Adam's axis is
        # unchanged, and the trainers still reject full-state GN continuation.
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


def restore_full_state_progress(metadata, flags, restored_step):
    """Recover reporting for legacy first-order full-state checkpoints.

    TrainState.step counts microsteps, including under Optax MultiSteps. This
    only recovers reporting; it does not claim an exact RNG/data continuation.
    """
    if not isinstance(metadata, dict) or metadata.get('step') != restored_step:
        raise ValueError('Full-state reporting requires metadata matching the restored step')
    saved_flags = metadata.get('flags', {})

    def dimensions(config):
        dataset = config.get('train_dataset', {})
        kind = dataset.get('type')
        if kind not in ('huggingface', 'json'):
            raise ValueError('Cannot reconstruct full-state reporting for this dataset type')
        dataset_config = dataset.get(f'{kind}_dataset', {})
        accumulation = config.get('optimizer', {}).get('accumulate_gradient_steps')
        batch_size = dataset_config.get('batch_size')
        seq_length = dataset_config.get('seq_length')
        if any(value is None or value <= 0 for value in
               (accumulation, batch_size, seq_length)):
            raise ValueError('Full-state metadata lacks accumulation, batch size or sequence length')
        return kind, accumulation, batch_size, seq_length

    saved_dimensions = dimensions(saved_flags)
    if saved_dimensions != dimensions(flags):
        raise ValueError('Full-state reporting requires unchanged dataset type, accumulation, '
                         'batch size and sequence length')
    kind, accumulation, batch_size, seq_length = saved_dimensions
    solved_tokens = restored_step * batch_size * seq_length
    completed_updates = restored_step // accumulation
    saved = metadata.get('training_progress')
    if saved is not None:
        progress = TrainingProgress.from_state_dict(saved)
        if (progress.phase_completed_updates != completed_updates
                or progress.phase_solve_tokens != solved_tokens):
            raise ValueError('Saved reporting counters do not match the restored optimizer step')
        return progress.state_dict()

    # Old trainer totals assumed a constant batch size throughout the phase.
    dataset_config = saved_flags['train_dataset'][f'{kind}_dataset']
    return TrainingProgress(
        step_offset=max(saved_flags.get('log_step_offset', -1), 0),
        token_offset=max(saved_flags.get('log_token_offset', -1), 0),
        phase_completed_updates=completed_updates,
        phase_solve_tokens=solved_tokens,
        dataset_total_tokens=dataset_config.get('tokens_count_at_start', 0) + solved_tokens,
        comparison_origin='full-state checkpoint metadata',
    ).state_dict()


def configure_wandb_run(run):
    run.define_metric("*", step_metric="total_tokens")
    for name in ("total_tokens", "cumulative_tokens", "completed_updates", "step", "global_step"):
        run.define_metric(name, hidden=True)
    for name in ("update_time_s", "train_time_s", "eval_time_s"):
        run.define_metric(name, step_metric="total_tokens", summary="last")


@dataclass
class ProcessTiming:
    """Process-local elapsed timing; deliberately absent from checkpoints."""
    clock: object = time.perf_counter
    train_time_s: float = 0.0
    eval_time_s: float = 0.0
    update_time_s: float = 0.0
    _started: float = None
    _pending_update_s: float = 0.0

    def start(self):
        if self._started is not None:
            raise RuntimeError("timing section already active")
        self._started = self.clock()

    def stop_train_interval(self, *, completed_update=True):
        elapsed = self._stop()
        self.train_time_s += elapsed
        self._pending_update_s += elapsed
        if completed_update:
            self.update_time_s = self._pending_update_s
            self._pending_update_s = 0.0
        return elapsed

    def stop_eval(self):
        elapsed = self._stop()
        self.eval_time_s += elapsed
        return elapsed

    def cancel(self):
        """Discard an interval that performed no work (for iterator exhaustion)."""
        self._started = None

    def metrics(self):
        return dict(update_time_s=self.update_time_s,
                    train_time_s=self.train_time_s, eval_time_s=self.eval_time_s)

    def _stop(self):
        if self._started is None:
            raise RuntimeError("timing section is not active")
        elapsed = self.clock() - self._started
        self._started = None
        return elapsed
