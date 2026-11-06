"""One native first-order update with paired data-loss measurements."""
import copy
import json
import os
import subprocess

import jax
import numpy as np
from flax.traverse_util import flatten_dict

from EasyLM.update_geometry import fingerprint_leaves


def validate_config(flags, model):
    """Validate finalized model settings before loading data or checkpoints."""
    def require(condition, message):
        if not condition:
            raise ValueError('batch_transfer_probe: ' + message)

    optimizer = flags['optimizer']
    require(optimizer['type'] == 'adamw', 'requires optimizer.type=adamw')
    require(optimizer['accumulate_gradient_steps'] == 1, 'requires accumulation=1')
    require(not optimizer['adamw_optimizer']['multiply_by_parameter_scale'],
            'requires ordinary AdamW (multiply_by_parameter_scale=False)')
    require(not flags['weight_average'], 'requires weight_average=False')
    require(flags['outer_loop_method'] in (None, 'None'), 'requires outer_loop_method=None')
    # This trainer calls W&B history reuse "replay"; diagnostics need a new run.
    require(not flags['wandb_run_id'], 'requires replay disabled (empty wandb_run_id)')
    require(flags['start_step'] == 0, 'requires start_step=0; optimizer step comes from state')
    dataset = flags['train_dataset']
    require(dataset['type'] == 'huggingface', 'requires raw Hugging Face streaming')
    hf = dataset['huggingface_dataset']
    require(not hf['pretokenized_dataset_dir'] and hf['streaming'],
            'requires raw Hugging Face streaming, without pretokenized data')
    require(hf['tokens_count_at_start'] == 0, 'requires tokens_count_at_start=0')
    for name in ('embedding_dropout', 'feedforward_dropout', 'attention_dropout',
                 'residue_dropout', 'fcm_min_ratio', 'fcm_max_ratio'):
        require(model[name] == 0, f'requires finalized model {name}=0')


def fresh_pair(factory, config, tokenizer):
    """Exactly the first and next batch of an independent, unrestored loader."""
    diagnostic_dataset = factory.load_dataset(copy.deepcopy(config), tokenizer)
    iterator = iter(diagnostic_dataset)
    try:
        update_batch, _ = next(iterator)
        transfer_batch, _ = next(iterator)
    except StopIteration as error:
        raise ValueError('batch_transfer_probe requires two complete training batches') from error
    return update_batch, transfer_batch


def fingerprint(tree, gathers):
    """All hosts gather every sorted leaf, keeping only one host leaf at a time.

    Gathers must replicate global arrays before host transfer, including on
    nonwriters. Never device_get a possibly non-addressable input shard.
    """
    flat_gathers = flatten_dict(gathers)
    return fingerprint_leaves((path, flat_gathers[path](value))
                              for path, value in sorted(flatten_dict(tree).items()))


def measure(train_state, train_rng, update_batch, transfer_batch, *,
            train_step, eval_step, param_gathers, batch_gathers, seed):
    """Consume donated state/RNG once; never access their old buffers afterward."""
    before_step = int(jax.device_get(train_state.step))
    captures = dict(base_parameters=fingerprint(train_state.params, param_gathers),
                    update_batch=fingerprint(update_batch, batch_gathers),
                    transfer_batch=fingerprint(transfer_batch, batch_gathers))
    # Construct independent buffers, without splitting/advancing the native RNG.
    diagnostic_rng = jax.random.PRNGKey(seed ^ 0xB47C)
    diagnostic_rng, transfer_before = eval_step(
        train_state.params, diagnostic_rng, transfer_batch)
    # Finish this read before the native call donates pre-update parameters.
    jax.block_until_ready((diagnostic_rng, transfer_before))
    train_state, train_rng, train_metrics = train_step(train_state, train_rng, update_batch)
    diagnostic_rng, update_after = eval_step(train_state.params, diagnostic_rng, update_batch)
    diagnostic_rng, transfer_after = eval_step(train_state.params, diagnostic_rng, transfer_batch)
    jax.block_until_ready((train_state, train_rng, diagnostic_rng, train_metrics,
                           update_after, transfer_after))
    after_step = int(jax.device_get(train_state.step))
    if after_step != before_step + 1:
        raise ValueError('batch_transfer_probe must apply exactly one optimizer update')
    losses = dict(loss_update_before=train_metrics['loss'],
                  loss_update_after=update_after['eval_loss'],
                  loss_transfer_before=transfer_before['eval_loss'],
                  loss_transfer_after=transfer_after['eval_loss'])
    losses = {name: float(jax.device_get(value)) for name, value in losses.items()}
    if not all(np.isfinite(value) for value in losses.values()):
        raise ValueError(f'batch_transfer_probe has nonfinite data losses: {losses}')
    losses.update(improvement_update=losses['loss_update_before'] - losses['loss_update_after'],
                  improvement_transfer=losses['loss_transfer_before'] - losses['loss_transfer_after'])
    if not all(np.isfinite(value) for value in losses.values()):
        raise ValueError('batch_transfer_probe has nonfinite loss decreases')
    metrics = {'batch_transfer/' + name: value for name, value in losses.items()}
    result = dict(format_version=1, method='first_order_adamw',
                  optimizer_step_before=before_step, optimizer_step_after=after_step,
                  data_mode='fresh_first_two_training_batches',
                  data_interpretation='Controlled checkpoint probe on a restarted training pair; '
                      'not continuation of the saved stream. B may have appeared in training history.',
                  metrics=metrics, **captures)
    return train_state, train_rng, result


def repository_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
            cwd=os.path.dirname(__file__), stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def complete(result, output_dir, *, writer, open_file, barrier, wandb):
    """Only the writer publishes; every host joins the completion barrier."""
    path = os.path.join(output_dir, 'batch_transfer', 'metadata.json')
    # measure has already blocked and checked losses; JSON rejects other NaNs.
    payload = json.dumps(dict(result, completed=True), indent=2, allow_nan=False)
    if writer:
        if not path.startswith('gs://'):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        wandb.log(result['metrics'], step=result['optimizer_step_after'], commit=True)
        with open_file(path, 'w') as stream:
            stream.write(payload + '\n')
    barrier('batch_transfer_complete')
    if writer:
        print(f'BATCH_TRANSFER_COMPLETE path={path}', flush=True)
    wandb.finish()
