"""Validation for complete, reproducible raw-HF Adam checkpoints."""
import os
import re
from copy import deepcopy


def normalized_dataset(config):
    config = deepcopy(dict(config))
    hf = dict(config.get('huggingface_dataset', {}))
    hf.setdefault('repeat_first_n_batches', 0)
    config['huggingface_dataset'] = hf
    return config


def validate_replay_config(flags):
    """Limit replay to the first-order, fixed-batch raw-HF training path."""
    train = normalized_dataset(flags.get('train_dataset', {}))
    hf = train['huggingface_dataset']
    repeat = hf['repeat_first_n_batches']
    if normalized_dataset(flags.get('eval_dataset', {}))['huggingface_dataset']['repeat_first_n_batches']:
        raise ValueError('Validation replay must remain disabled')
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 0:
        raise ValueError('repeat_first_n_batches must be a nonnegative integer')
    if not repeat:
        return
    if (train.get('type', 'huggingface') != 'huggingface'
            or hf.get('pretokenized_dataset_dir', '')
            or hf.get('tokens_count_at_start', 0) != 0
            or flags.get('optimizer', {}).get('type') != 'adamw'
            or flags['optimizer'].get('accumulate_gradient_steps', 1) != 1
            or flags.get('outer_loop_method', 'None') != 'None'
            or flags.get('weight_average', False)
            or flags.get('start_step', 0) != 0
            or flags.get('log_step_offset', -1) not in (-1, 0)
            or flags.get('log_token_offset', -1) not in (-1, 0)):
        raise ValueError('Replay requires raw-HF first-order Adam, accumulation=1, fixed batches and zero-origin accounting')
    if flags.get('load_checkpoint') and not flags['load_checkpoint'].startswith('trainstate::'):
        raise ValueError('Replay continuation requires a full trainstate:: Adam checkpoint')
    if flags.get('load_dataset_state') and not flags.get('load_checkpoint', '').startswith('trainstate::'):
        raise ValueError('Replay dataset restoration requires its full Adam checkpoint')


def resume_companion_paths(checkpoint_path):
    match = re.fullmatch(r"streaming_train_state(_[0-9]+)?", os.path.basename(checkpoint_path))
    if match is None:
        raise ValueError("Full resume requires a streaming_train_state checkpoint")
    suffix = match.group(1) or ""
    directory = os.path.dirname(checkpoint_path)
    return {name: os.path.join(directory, f"{name}{suffix}.pkl")
            for name in ("metadata", "dataset", "complete")}


def validate_resume_metadata(metadata, complete, flags, dataset_state=None):
    if not isinstance(metadata, dict) or metadata.get("resume_version") != 1:
        raise ValueError("Checkpoint lacks complete RNG/data resume metadata")
    if (not metadata.get("optimizer_saved")
            or metadata.get("checkpoint_float_dtype") != "fp32"):
        raise ValueError("Exact Adam resume requires optimizer state saved in fp32")
    if "train_rng" not in metadata or "step" not in metadata:
        raise ValueError("Checkpoint lacks RNG or completed-step count")
    if (not isinstance(complete, dict) or complete.get("resume_version") != 1
            or not metadata.get("snapshot_id")
            or complete.get("snapshot_id") != metadata["snapshot_id"]
            or complete.get("step") != metadata["step"]):
        raise ValueError("Checkpoint is incomplete or its completion record does not match")
    saved = deepcopy(dict(metadata['flags']))
    flags = deepcopy(dict(flags))
    for key in ('train_dataset', 'eval_dataset'):
        saved[key] = normalized_dataset(saved.get(key, {}))
        flags[key] = normalized_dataset(flags.get(key, {}))
    validate_replay_config(flags)
    old_hf = saved['train_dataset']['huggingface_dataset']
    new_hf = flags['train_dataset']['huggingface_dataset']
    transition = old_hf['repeat_first_n_batches'] == 0 and new_hf['repeat_first_n_batches'] > 0
    if transition:
        validate_dataset_snapshot(dataset_state, metadata)
        if (saved.get('optimizer', {}).get('type') != 'adamw'
                or saved['optimizer'].get('accumulate_gradient_steps', 1) != 1
                or saved.get('dtype') != 'fp32' or saved.get('param_dtype') != 'fp32'
                or flags.get('dtype') != 'fp32' or flags.get('param_dtype') != 'fp32'
                or old_hf.get('tokens_count_at_start') != 0
                or saved.get('weight_average', False)
                or saved.get('start_step', 0) != 0
                or saved.get('log_step_offset', -1) not in (-1, 0)
                or saved.get('log_token_offset', -1) not in (-1, 0)):
            raise ValueError('Replay parent must be full FP32 Adam with zero-origin accounting')
        budget = new_hf['repeat_first_n_batches'] * new_hf['batch_size'] * new_hf['seq_length']
        if not 0 <= dataset_state['metadata']['dataset_total_tokens'] <= budget:
            raise ValueError('Replay parent consumed more than the first replay epoch')
        progress = metadata.get('training_progress')
        if progress is not None and (
                progress.get('step_offset') != 0 or progress.get('token_offset') != 0
                or progress.get('phase_completed_updates') != metadata['step']
                or progress.get('phase_solve_tokens') != dataset_state['metadata']['dataset_total_tokens']
                or progress.get('dataset_total_tokens') != dataset_state['metadata']['dataset_total_tokens']
                or progress.get('phase_linesearch_tokens', 0) != 0
                or progress.get('phase_skipped_tokens', 0) != 0):
            raise ValueError('Replay parent reporting counters must match zero-origin Adam updates')
        parent_dir = os.path.dirname(flags.get('load_checkpoint', '::').split('::', 1)[-1])
        output = os.path.join(flags.get('output_dir', ''), flags.get('experiment_id', ''))
        saved_output = os.path.join(saved.get('output_dir', ''), saved.get('experiment_id', ''))
        if (not flags.get('experiment_id') or flags.get('wandb_run_id')
                or os.path.normpath(output) in (os.path.normpath(parent_dir), os.path.normpath(saved_output))):
            raise ValueError('Initial replay fork requires a new output experiment directory and new W&B run (empty wandb_run_id)')
        if flags['total_steps'] < saved['total_steps']:
            raise ValueError('Replay transition may only increase total_steps')
        if flags.get('target_loss') != saved.get('target_loss') and flags.get('target_loss') != 0:
            raise ValueError('Replay transition may only change target_loss to 0')
        old_hf['repeat_first_n_batches'] = new_hf['repeat_first_n_batches']
        saved['total_steps'] = flags['total_steps']
        saved['target_loss'] = flags.get('target_loss')
    keys = ("seed", "mesh_dim", "dtype", "param_dtype", "total_steps", "tokenizer",
            "llama", "optimizer", "train_dataset", "train_dataset_batch_size",
            "eval_dataset", "eval_steps", "eval_freq", "log_freq", "weight_average",
            "weight_average_decay", "target_loss", "load_llama_config",
            "update_llama_config", "outer_loop_method")
    changed = [key for key in keys if saved.get(key) != flags.get(key)]
    if changed:
        raise ValueError("Resume changes trajectory settings: " + ", ".join(changed))
    if flags.get("eval_steps", 0) > 0 and flags.get("eval_freq", 0) != 0:
        eval_config = flags.get("eval_dataset", {})
        # Match DatasetFactory defaults. Only raw HF validation restarts from
        # the same data on every evaluation; other loaders retain mutable state
        # that these exact-resume checkpoints do not save.
        if (eval_config.get("type", "huggingface") != "huggingface"
                or eval_config.get("huggingface_dataset", {}).get("pretokenized_dataset_dir", "")):
            raise ValueError("Exact Adam resume with validation requires a raw Hugging Face "
                             "evaluation dataset; pretokenized/JSON validation state is not checkpointed")
    return transition


def validate_dataset_snapshot(state, metadata):
    if not isinstance(state, dict) or state.get("packed_state_version") != 1:
        raise ValueError("Exact resume requires the saved token-packing remainder")
    required = ('config', 'text_processor_config', 'tokenizer_name',
                'tokenizer_vocab_size', 'examples_consumed', 'token_buffer',
                'loss_mask_buffer', 'metadata')
    if (any(key not in state for key in required)
            or not isinstance(state['metadata'], dict)
            or 'dataset_total_tokens' not in state['metadata']
            or len(state['token_buffer']) != len(state['loss_mask_buffer'])):
        raise ValueError('Exact resume requires a complete packed dataset snapshot')
    if (state.get("training_step") != metadata["step"]
            or state.get("snapshot_id") != metadata["snapshot_id"]):
        raise ValueError("Dataset and model metadata belong to different snapshots")
    cfg = metadata["flags"]["train_dataset"]["huggingface_dataset"]
    snapshot_config = dict(state['config'])
    snapshot_config.setdefault('repeat_first_n_batches', 0)
    expected_config = normalized_dataset(metadata['flags']['train_dataset'])['huggingface_dataset']
    if snapshot_config != expected_config:
        raise ValueError('Dataset config and model metadata belong to different snapshots')
    expected_tokens = (cfg["tokens_count_at_start"] + metadata["step"]
                       * cfg["batch_size"] * cfg["seq_length"])
    if state["metadata"]["dataset_total_tokens"] != expected_tokens:
        raise ValueError("Packed dataset token count does not match completed updates")


def validate_branch_parent(metadata, complete, dataset, step_offset, token_offset):
    """Validate a params-only branch without restoring the parent's optimizer."""
    if step_offset < 0 or token_offset < 0:
        raise ValueError("Parameter-only continuation requires explicit reporting offsets")
    if (not isinstance(metadata, dict) or not isinstance(complete, dict)
            or not isinstance(dataset, dict)):
        raise ValueError("Parameter-only branch checkpoint companions are missing")
    snapshot_id = metadata.get("snapshot_id")
    if (not snapshot_id or complete.get("snapshot_id") != snapshot_id
            or dataset.get("snapshot_id") != snapshot_id):
        raise ValueError("Parameter-only branch companions belong to different snapshots")
    if (metadata.get("step") != step_offset
            or complete.get("step") != step_offset
            or dataset.get("training_step") != step_offset):
        raise ValueError("log_step_offset does not match the parent checkpoint bundle")
    cursor = dataset.get("metadata", {}).get("dataset_total_tokens")
    if cursor != token_offset:
        raise ValueError("log_token_offset does not match the packed parent cursor")
