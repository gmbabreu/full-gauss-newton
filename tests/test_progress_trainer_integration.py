"""Source-level regression gates for progress integration in heavyweight trainers."""
from pathlib import Path
import re

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


def test_timers_start_before_fetch_and_synchronize_before_stop():
    adam_loop = ADAM[ADAM.index('for step in step_counter:'):]
    assert adam_loop.index('timing.start()') < adam_loop.index('next(train_iterator)')
    assert adam_loop.index('jax.block_until_ready') < adam_loop.index('timing.stop_train_interval')
    gn_loop = GN[GN.index('for step in step_counter:'):]
    assert gn_loop.index('timing.start()') < gn_loop.index("pull_training_batch('skipped')")
    ready = gn_loop.index('jax.block_until_ready(live_results)')
    assert ready < gn_loop.index('timing.stop_train_interval(completed_update=True)', ready)


def test_timing_config_and_terminal_summary_are_process_local():
    for source in (ADAM, GN):
        assert "'timing_scope': 'current_process'" in source
        assert "'timing_includes_first_use_compilation': True" in source
        assert 'terminal_record = progress.record(' in source
        assert 'timing.stop_eval()' in source


def test_gn_interrupted_work_is_preserved_without_completing_update():
    assert 'timing.cancel()' not in GN
    assert GN.count('timing.stop_train_interval(completed_update=False)') == 3
    assert GN.count('jax.block_until_ready') >= 7


def test_final_timing_summary_is_outside_terminal_eval_condition():
    for source in (ADAM, GN):
        terminal_if = source.rindex('if FLAGS.eval_freq != 0 and FLAGS.eval_steps > 0:')
        summary = source.rindex("wandb.run.summary[f'terminal_{name}']")
        finish = source.index('\n    wandb.finish()', terminal_if)
        # The summary loop is dedented to the same trainer scope as the
        # conditional, so it runs whether or not terminal evaluation is enabled.
        assert terminal_if < summary < finish
        assert re.search(
            r"\n        for name, value in terminal_record\.items\(\):\n"
            r"            wandb\.run\.summary\[f'terminal_\{name\}'\]", source)
