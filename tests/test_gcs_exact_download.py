from pathlib import Path
from unittest.mock import Mock, patch
from EasyLM.gcs_utils import load_from_gcs

def test_exact_file_wins(tmp_path):
    exact, sibling = Mock(), Mock(); exact.name='run/state'; sibling.name='run/state_900'
    bucket=Mock(); bucket.list_blobs.return_value=[sibling, exact]
    exact.download_to_filename.side_effect=lambda p: Path(p).write_bytes(b'ok')
    with patch('EasyLM.gcs_utils.storage.Client') as client:
        client.return_value.bucket.return_value=bucket
        target=str(tmp_path/'model'); assert load_from_gcs('gs://bucket/run/state',target)==target
    assert Path(target).read_bytes()==b'ok'; sibling.download_to_filename.assert_not_called()

def test_directory_download(tmp_path):
    blobs=[Mock(),Mock()]; blobs[0].name='data/a'; blobs[1].name='data/nested/b'
    for blob in blobs: blob.download_to_filename.side_effect=lambda p: Path(p).write_bytes(b'x')
    bucket=Mock(); bucket.list_blobs.return_value=blobs
    with patch('EasyLM.gcs_utils.storage.Client') as client:
        client.return_value.bucket.return_value=bucket
        assert load_from_gcs('gs://bucket/data',str(tmp_path/'out')).endswith('/')
    assert (tmp_path/'out/a').read_bytes()==b'x'; assert (tmp_path/'out/nested/b').read_bytes()==b'x'
