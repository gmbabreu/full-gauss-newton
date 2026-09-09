import pytest
from EasyLM.training_progress import ProcessTiming, TrainingProgress, resolve_progress
from EasyLM.training_resume import validate_branch_parent

class Tokens:
    def __init__(self, size):
        self.size = size


def batch(n):
    return {'input_tokens': Tokens(n * 1024)}
def test_adam300_gn_b640_and_roundtrip():
    p=resolve_progress(300,196608000,branch=True)
    assert (p.record(-1)['step'],p.record(-1)['total_tokens'])==(300,196608000)
    p.charge('skipped',batch(640),{'dataset_total_tokens':197263360})
    p.charge('solve',batch(640),{'dataset_total_tokens':197918720})
    p.charge('linesearch',batch(640),{'dataset_total_tokens':198574080}); p.complete_update()
    r=p.record(0); assert (r['step'],r['total_tokens'],r['dataset_total_tokens'],r['phase_skipped_tokens'])==(301,197263360,198574080,655360)
    assert TrainingProgress.from_state_dict(p.state_dict()).record(0)==r
def test_b720_k_and_candidates_do_not_multiply():
    for k in (1,50):
        p=resolve_progress(300,196608000,branch=True); p.charge('skipped',batch(720)); p.charge('solve',batch(720))
        p.charge('linesearch',batch(720)); p.complete_update(); r=p.record(0)
        assert (r['total_tokens'],r['dataset_total_tokens'])==(197345280,198819840)
def test_offsets_conflict_and_accumulation_boundary():
    p=TrainingProgress(); p.charge('solve',batch(2)); assert p.completed_updates==0
    p.complete_update(); assert p.completed_updates==1
    state=p.state_dict()
    with pytest.raises(ValueError,match='conflicts'): resolve_progress(9,-1,state)
    with pytest.raises(ValueError,match='both'): resolve_progress(3,-1,branch=True)


@pytest.mark.parametrize('parent_step', [300, 600])
@pytest.mark.parametrize('linesearch_batches', [0, 1, 5])
def test_solve_axis_excludes_search_and_skips(parent_step, linesearch_batches):
    tokens = 640 * 1024
    prefix = parent_step * tokens
    p = resolve_progress(parent_step, prefix, branch=True)
    p.charge('skipped', batch(640))
    p.charge('solve', batch(640))
    for _ in range(linesearch_batches):
        p.charge('linesearch', batch(640))
    p.complete_update()
    row = p.record(0)
    assert row['step'] == parent_step + 1
    assert row['total_tokens'] == row['cumulative_tokens'] == prefix + tokens
    assert row['phase_linesearch_tokens'] == linesearch_batches * tokens
    assert row['phase_skipped_tokens'] == tokens
    assert row['dataset_total_tokens'] == prefix + (2 + linesearch_batches) * tokens
    assert row['progress_schema_version'] == 3


def test_distinct_solve_batches_and_unlogged_updates_accumulate():
    p = TrainingProgress()
    for _ in range(2):
        for size in (640, 128):
            p.charge('solve', batch(size))
        p.charge('linesearch', batch(640))
        p.complete_update()
    row = p.record(1)
    assert row['completed_updates'] == 2
    assert row['total_tokens'] == 2 * (640 + 128) * 1024
    assert row['phase_linesearch_tokens'] == 2 * 640 * 1024


def test_v2_adam_resume_migrates_without_changing_coordinates():
    p = TrainingProgress(phase_completed_updates=600,
                         phase_solve_tokens=393216000,
                         dataset_total_tokens=393216000)
    old_state = p.state_dict()
    old_state.update(progress_schema_version=2,
                     token_convention='distinct global input-token positions used by solve or line search')
    restored = resolve_progress(saved_state=old_state)
    assert restored.record(599)['total_tokens'] == 393216000
    assert restored.completed_updates == 600
    assert old_state['progress_schema_version'] == 2
    assert restored.state_dict()['progress_schema_version'] == 3
    restored.charge('solve', batch(640))
    restored.complete_update()
    assert restored.record(600)['total_tokens'] == 393871360
    with pytest.raises(ValueError, match='unsupported'):
        TrainingProgress.from_state_dict(dict(old_state, progress_schema_version=99))


def test_params_only_parent_bundle_validation():
    metadata = {'step': 300, 'snapshot_id': 'same'}
    complete = {'step': 300, 'snapshot_id': 'same'}
    dataset = {'training_step': 300, 'snapshot_id': 'same',
               'metadata': {'dataset_total_tokens': 196608000}}
    validate_branch_parent(metadata, complete, dataset, 300, 196608000)
    with pytest.raises(ValueError, match='different snapshots'):
        validate_branch_parent(metadata, complete, dict(dataset, snapshot_id='bad'),
                               300, 196608000)
    with pytest.raises(ValueError, match='log_token_offset'):
        validate_branch_parent(metadata, complete, dataset, 300, 1)


def test_process_local_training_and_evaluation_timing():
    readings = iter((0, 3, 3, 5, 5, 10, 10, 14, 14, 15))
    timing = ProcessTiming(clock=lambda: next(readings))
    timing.start(); timing.stop_eval()
    timing.start(); timing.stop_train_interval(completed_update=True)
    timing.start(); timing.stop_eval()
    assert timing.metrics() == {
        'update_time_s': 2, 'train_time_s': 2, 'eval_time_s': 8}
    timing.start(); timing.stop_train_interval(completed_update=True)
    assert timing.metrics() == {
        'update_time_s': 4, 'train_time_s': 6, 'eval_time_s': 8}
    timing.start(); timing.stop_eval()
    assert timing.metrics() == {
        'update_time_s': 4, 'train_time_s': 6, 'eval_time_s': 9}


def test_accumulation_timing_is_pending_until_update():
    readings = iter((0, 1, 1, 3))
    timing = ProcessTiming(clock=lambda: next(readings))
    timing.start(); timing.stop_train_interval(completed_update=False)
    assert timing.update_time_s == 0 and timing.train_time_s == 1
    timing.start(); timing.stop_train_interval(completed_update=True)
    assert timing.update_time_s == 3 and timing.train_time_s == 3
