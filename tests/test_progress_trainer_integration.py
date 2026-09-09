"""Source-level regression gates for progress integration in heavyweight trainers."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADAM = (ROOT / 'EasyLM/models/llama/llama_train.py').read_text()
GN = (ROOT / 'EasyLM/models/llama/llama_train_gn.py').read_text()


def test_initial_eval_copies_rng_before_donating_eval_call():
    expected = "initial_eval_rng = jax.tree.map(lambda x: x.copy(), sharded_rng)"
    assert expected in ADAM
    assert expected in GN
    assert "initial_eval_rng = sharded_rng" not in ADAM
    assert "initial_eval_rng = sharded_rng" not in GN


def test_adam_reporting_retains_local_evaluation_cadence():
    assert "if applied_update and step % FLAGS.log_freq == 0:" in ADAM
    assert "progress.phase_completed_updates % FLAGS.log_freq" not in ADAM
    assert "step % FLAGS.eval_freq == 0" in ADAM


def test_gn_line_search_does_not_overwrite_solver_loss():
    assert GN.count('"ls_baseline_loss": baseline_loss') == 3
    assert '"loss": baseline_loss' not in GN
    # Only initial and completed outer rows commit W&B history.
    active_logs = [line for line in GN.splitlines()
                   if 'wandb.log(' in line and not line.lstrip().startswith('#')]
    assert len(active_logs) == 2


def test_gn_restore_is_validated_or_rejected():
    assert "if init_checkpoint_path.startswith('trainstate::'):" in GN
    assert 'validate_branch_parent(' in GN
    assert "branch_parent_metadata = load_parent_companion('metadata')" in GN
    assert "branch_parent_complete = load_parent_companion('complete')" in GN


def test_terminal_summary_uses_current_progress_boundary():
    expected = "progress.phase_completed_updates - 1"
    assert expected in ADAM
    assert expected in GN
    assert '{"global_step": start_step}' not in ADAM
    assert '{"global_step": start_step}' not in GN
