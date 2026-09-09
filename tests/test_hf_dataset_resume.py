import pickle
from unittest.mock import patch
import numpy as np
import pytest
from datasets import IterableDataset
from EasyLM.data import HuggingfaceDataset, TextProcessor
class Tokenizer:
    name_or_path='integer'; bos_token_id=-1; eos_token_id=-2
    def __len__(self): return 1000000
    def encode(self,text,add_special_tokens=False): return [int(x) for x in text.split()]
def examples():
    lengths=[0,1,2,4,12,13,14,63,171]
    for i in range(480): yield {'prefix':str(i*1000),'text':' '.join(str(i*1000+j+1) for j in range(lengths[i%9]))}
def make_dataset(**kw):
    cfg=dict(path='local',name='',split='train',streaming=True,batch_size=3,seq_length=4,shuffle_data=True,shuffle_seed=42,shuffle_buffer_size=7); cfg.update(kw)
    tok=Tokenizer(); proc=TextProcessor(dict(fields='[prefix],text',add_bos_token=False),tok)
    with patch('EasyLM.data.load_dataset',return_value=IterableDataset.from_generator(examples)): return HuggingfaceDataset(cfg,tok,proc)
def equal(a,b):
    for k in a: np.testing.assert_array_equal(a[k],b[k])
def test_resume_twice_and_repeat():
    ds=make_dataset(); it=iter(ds); first=next(it)[0]
    for _ in range(899): next(it)
    saved=pickle.loads(pickle.dumps(ds.get_state_dict())); expected=[next(it)[0] for _ in range(60)]
    resumed=make_dataset(); resumed.load_state_dict(saved); ri=iter(resumed)
    for b in expected[:23]: equal(next(ri)[0],b)
    saved2=pickle.loads(pickle.dumps(resumed.get_state_dict())); again=make_dataset(); again.load_state_dict(saved2); ai=iter(again)
    for b in expected[23:]: equal(next(ai)[0],b)
    equal(next(iter(ds))[0],first); equal(next(iter(resumed))[0],expected[0])
def test_config_mismatch_and_copy():
    ds=make_dataset(); it=iter(ds); next(it); saved=ds.get_state_dict(); frozen=pickle.loads(pickle.dumps(saved)); next(it); assert saved==frozen
    with pytest.raises(ValueError,match='shuffle_seed'): make_dataset(shuffle_seed=9).load_state_dict(saved)
