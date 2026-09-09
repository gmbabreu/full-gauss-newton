"""Rebatch a serialized packed stream without changing its token sequence."""
import ast
import copy
from pathlib import Path
import pickle
from types import SimpleNamespace
from unittest.mock import patch

from datasets import IterableDataset
import numpy as np
import pytest

from EasyLM.data import HuggingfaceDataset, TextProcessor


class IntegerTokenizer:
    name_or_path = 'rebatch-test'
    bos_token_id = -1
    eos_token_id = -2

    def __len__(self):
        return 10_000_000

    def encode(self, text, add_special_tokens=False):
        return [int(x) for x in text.split()]


def examples():
    sizes = [0, 1, 2, 4, 12, 13, 14, 63, 171]
    for i in range(2000):
        yield {
            'prefix': str(i * 1000),
            'text': ' '.join(str(i * 1000 + j + 1) for j in range(sizes[i % len(sizes)])),
        }


def make_dataset(**overrides):
    cfg = dict(path='local-rebatch-test', name='', split='train', streaming=True,
               batch_size=640, seq_length=4, shuffle_data=True,
               shuffle_seed=42, shuffle_buffer_size=7)
    cfg.update(overrides)
    tokenizer = IntegerTokenizer()
    processor = TextProcessor(dict(fields='[prefix],text', add_bos_token=False), tokenizer)
    with patch('EasyLM.data.load_dataset', return_value=IterableDataset.from_generator(examples)):
        return HuggingfaceDataset(cfg, tokenizer, processor)


def snapshot():
    data = make_dataset()
    iterator = iter(data)
    for _ in range(3):
        next(iterator)
    return pickle.loads(pickle.dumps(data.get_state_dict()))


def test_640_to_720_matches_independent_flat_stream():
    data = make_dataset()
    encoded = [data.text_processor(example) for example in data.dataset]
    tokens = np.concatenate([x[0] for x in encoded])
    masks = np.concatenate([x[1] for x in encoded]).astype(np.float32)
    iterator = iter(data)
    for _ in range(3):
        next(iterator)
    saved = pickle.loads(pickle.dumps(data.get_state_dict()))
    frozen = copy.deepcopy(saved)
    gn = make_dataset(batch_size=720)
    gn.load_state_dict(saved, allow_batch_size_change=True)
    assert gn.config.batch_size == 720 and saved == frozen
    gn_iterator = iter(gn)
    actual = [next(gn_iterator)[0] for _ in range(8)]
    start = 3 * 640 * 4
    count = 8 * 720 * 4
    expected = {
        'input_tokens': tokens[start:start + count],
        'target_tokens': tokens[start + 1:start + count + 1],
        'loss_masks': masks[start + 1:start + count + 1],
    }
    for name in expected:
        flat = np.concatenate([batch[name].ravel() for batch in actual])
        np.testing.assert_array_equal(flat, expected[name])
    assert gn.get_state_dict()['metadata']['dataset_total_tokens'] == start + count

    saved_again = pickle.loads(pickle.dumps(gn.get_state_dict()))
    continued = make_dataset(batch_size=720)
    continued.load_state_dict(saved_again)
    expected_next = next(gn_iterator)[0]
    actual_next = next(iter(continued))[0]
    for name in expected_next:
        np.testing.assert_array_equal(actual_next[name], expected_next[name])
    first_again = next(iter(gn))[0]
    for name in first_again:
        np.testing.assert_array_equal(first_again[name], actual[0][name])
    print('640->720: 23040 next input/target/mask entries exact; second resume exact')


def test_opt_in_changes_only_batch_size():
    saved = snapshot()
    with pytest.raises(ValueError, match='batch_size'):
        make_dataset(batch_size=720).load_state_dict(saved)
    for field, value in [('seq_length', 5), ('shuffle_seed', 9), ('path', 'other')]:
        with pytest.raises(ValueError, match=field):
            make_dataset(batch_size=720, **{field: value}).load_state_dict(
                saved, allow_batch_size_change=True)


def test_actual_gn_restore_block_local_and_gcs():
    path = Path(__file__).resolve().parents[1] / 'EasyLM/models/llama/llama_train_gn.py'
    source = path.read_text()
    start = source.index("    if FLAGS.load_dataset_state.startswith('gs://'):")
    end = source.index('    if FLAGS.eval_steps > 0:', start)
    import textwrap
    block = compile(ast.parse(textwrap.dedent(source[start:end])), str(path), 'exec')
    for remote in (False, True):
        saved = snapshot()
        events = []
        flags = SimpleNamespace(
            load_dataset_state='gs://test/dataset.pkl' if remote else '/tmp/state.pkl',
            tmp_dir='/tmp', tokenizer='fake', train_dataset=None,
            optimizer_type='muon', gauss_newton=True,
        )

        def download(remote_path, local_path):
            events.append('download')
            return local_path

        def load(path):
            assert not path.startswith('gs://')
            events.append('load')
            return copy.deepcopy(saved)

        namespace = dict(
            FLAGS=flags, load_from_gcs=download, os=__import__('os'),
            mlxu=SimpleNamespace(load_pickle=load),
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda _: None),
            DatasetFactory=SimpleNamespace(load_dataset=lambda *_: make_dataset(batch_size=720)),
            HuggingfaceDataset=HuggingfaceDataset,
            validate_branch_parent=lambda *args: None,
            branch_parent_metadata={}, branch_parent_complete={},
            init_checkpoint_path='trainstate_params::/tmp/streaming_train_state_300',
        )
        exec(block, namespace)
        assert events == (['download', 'load'] if remote else ['load'])
        assert namespace['dataset'].config.batch_size == 720
        assert namespace['dataset'].get_state_dict()['examples_consumed'] == saved['examples_consumed']
        namespace['init_checkpoint_path'] = 'trainstate::/tmp/streaming_train_state_300'
        with pytest.raises(ValueError, match='batch_size'):
            exec(block, namespace)
