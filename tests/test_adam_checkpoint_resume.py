import copy
from pathlib import Path
import jax.numpy as jnp
import pytest
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.training_resume import resume_companion_paths,validate_resume_metadata,validate_dataset_snapshot

def test_resume_validation():
    flags={'total_steps':3000,'optimizer':{'lr_decay_steps':3000},'eval_freq':1,'log_freq':1,'eval_steps':50,'train_dataset':{'huggingface_dataset':{'tokens_count_at_start':0,'batch_size':3,'seq_length':4}}}
    meta={'resume_version':1,'step':900,'snapshot_id':'id','optimizer_saved':True,'checkpoint_float_dtype':'fp32','train_rng':[1,2],'flags':copy.deepcopy(flags)}
    complete={'resume_version':1,'step':900,'snapshot_id':'id'}; data={'packed_state_version':1,'training_step':900,'snapshot_id':'id','metadata':{'dataset_total_tokens':10800}}
    validate_resume_metadata(meta,complete,flags); validate_dataset_snapshot(data,meta)
    assert resume_companion_paths('gs://b/streaming_train_state_900')['dataset']=='gs://b/dataset_900.pkl'
    with pytest.raises(ValueError,match='incomplete'): validate_resume_metadata(meta,dict(complete,snapshot_id='bad'),flags)
    bad=copy.deepcopy(flags); bad['eval_freq']=2
    with pytest.raises(ValueError,match='eval_freq'): validate_resume_metadata(meta,complete,bad)
def test_uploaded_temp_removed(monkeypatch):
    paths=[]
    def upload(local,remote): assert Path(local).is_file(); paths.append(local)
    monkeypatch.setattr(StreamingCheckpointer,'upload_to_gcs',upload)
    StreamingCheckpointer.save_train_state_to_file({'w':jnp.array([.3])},'gs://x/state',float_dtype='fp32')
    assert paths and not Path(paths[0]).exists()
