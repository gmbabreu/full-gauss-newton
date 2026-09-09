import pytest
from EasyLM.training_progress import TrainingProgress, resolve_progress

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
    r=p.record(0); assert (r['step'],r['total_tokens'],r['dataset_total_tokens'],r['phase_skipped_tokens'])==(301,197918720,198574080,655360)
    assert TrainingProgress.from_state_dict(p.state_dict()).record(0)==r
def test_b720_k_and_candidates_do_not_multiply():
    for k in (1,50):
        p=resolve_progress(300,196608000,branch=True); p.charge('skipped',batch(720)); p.charge('solve',batch(720))
        p.charge('linesearch',batch(720)); p.complete_update(); r=p.record(0)
        assert (r['total_tokens'],r['dataset_total_tokens'])==(198082560,198819840)
def test_offsets_conflict_and_accumulation_boundary():
    p=TrainingProgress(); p.charge('solve',batch(2)); assert p.completed_updates==0
    p.complete_update(); assert p.completed_updates==1
    state=p.state_dict()
    with pytest.raises(ValueError,match='conflicts'): resolve_progress(9,-1,state)
    with pytest.raises(ValueError,match='both'): resolve_progress(3,-1,branch=True)
