import pprint
from functools import partial

from google.cloud import storage

from tqdm import tqdm, trange
import numpy as np
import mlxu
import subprocess as sp

import timeit
import math
import os
import wandb
from libtpu.sdk import monitoring as tpu_monitoring
import copy

import jax
import jax.numpy as jnp
from jax import linearize, linear_transpose
from jax.experimental import multihost_utils
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from flax.training.train_state import TrainState
from transformers import AutoTokenizer
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.scipy.sparse.linalg import cg

import optax

from EasyLM.data import DatasetFactory, HuggingfaceDataset
from EasyLM.training_progress import ProcessTiming, configure_wandb_run, resolve_progress
from EasyLM import cg_resume, matrix_condition
from EasyLM.training_resume import resume_companion_paths, validate_branch_parent
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.optimizers import OptimizerFactory
from EasyLM.jax_utils import (
    JaxRNG, JaxDistributedConfig, next_rng, match_partition_rules,
    cross_entropy_loss_and_accuracy, global_norm, tree_dot, get_float_dtype_by_name,
    set_random_seed, average_metrics, make_shard_and_gather_fns,
    with_sharding_constraint, cross_entropy_loss_and_accuracy_with_weight_decay, CustomTrainState
)
from EasyLM.models.llama.llama_model import (
    LLaMAConfigurator, FlaxLLaMAForCausalLMModule
)
from EasyLM.gcs_utils import (
    load_ckpt_from_gcs, load_from_gcs, 
    upload_to_gcs, load_first_n_files_from_gcs, 
    modify_dataset_info_gcs, modify_state_json_gcs
)

FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    seed=42,
    mesh_dim='1,-1,1',
    dtype='fp32',
    param_dtype='fp32',
    total_steps=10000,
    load_llama_config='',
    update_llama_config='',
    load_checkpoint='',
    cg_resume_state='',  # Internal: complete CG bundle selected by the launcher.
    load_dataset_state='',
    log_freq=50,
    log_step_offset=-1,
    log_token_offset=-1,
    log_time_offset_s=0.0,
    log_initial_eval=False,
    log_inner_steps=False,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_freq=0,
    eval_steps=0,
    gradient_accumulation_steps=1,   # Dead flag
    inner_loop_iter=100,
    tokenizer='openlm-research/open_llama_3b_v2',
    train_dataset_batch_size=8,
    train_batch_growth_interval=0,
    train_batch_growth_increment=16,
    train_batch_max=1024,
    train_dataset=DatasetFactory.get_default_config(),
    eval_dataset=DatasetFactory.get_default_config(),
    optimizer=OptimizerFactory.get_default_config(),
    checkpointer=StreamingCheckpointer.get_default_config(),
    llama=LLaMAConfigurator.get_default_config(),
    # logger=mlxu.WandBLogger.get_default_config(),
    log_all_worker=False,
    jax_distributed=JaxDistributedConfig.get_default_config(),
    outer_loop_method='replace',
    lr_sched='cosine',
    inner_loop_lr=0.001,
    inner_loop_wd=0.0,
    end_lr=0.0,
    global_warmup=0.2,
    inner_loop_warmup=0.0,

    optimizer_type='adamw',
    inner_b1=0.9,
    inner_b2=0.999,
    inner_clip_gradient=0.0,
    optimizer_wd=0.0,
    outer_weight_decay=0.0,  # fractional outer shrink; independent of line search
    parameter_wd=0.0,  # Dead flag

    wandb_run_id='',
    start_tokens=0,

    wandb_project='',
    wandb_entity='harvardml',
    wandb_dir='/n/netscratch/kempner_barak_lab/Lab/nabreu/SOO-LM/experiment_output/open_llama_7b',
    output_dir='',
    notes='',
    logger=mlxu.WandBLogger.get_default_config(),
    experiment_id='',
    
    # GCS specific flags
    gcs_num_train_files_to_download=300,
    tmp_dir='/tmp',

    weight_average=False,
    weight_average_decay=0.99,
    load_ema_checkpoint='',
    linesearch=False,
    ls_range=5,
    normalize_step=False,
    single_batch_inner=False,
    ls_lambdas='',
    fixed_step_size=0.0,
    ls_eval_batches=0,  # 0 means: default to inner_loop_iter
    outer_momentum_beta=0.0,
    armijo_linesearch=False,
    adaptive_inner_loop=False,
    armijo_alpha=0.5,
    armijo_beta=0.5,
    armijo_init_step=1.0,

    gauss_newton=False,
    redo_gn=0,
    reset_start=False,

    target_loss=0.0,

    patience=1,

    cg_tol=1e-5,   # Relative Residual Tolerance for CG
    cg_atol=0.0,    # Absolute residual tolerance for CG
    cg_maxiter=100, # Maximum number of CG iterations
    cg_interpolation_lambda=1.0,
    cg_lambda_batch_denominator=0.0,  # Positive: global solve sequences / denominator.
    cg_lambda_final=-1.0,
    cg_lambda_ramp_steps=0,
    cg_n_micro=1,   # microbatches for CG G; 1 = no microbatching (default, backward-compatible)
    # Observational spectral diagnostics.  They run only at the requested
    # cadence and never alter the solve operator or effective lambda.
    condition_log=False,
    condition_every=50,
    condition_top_maxiter=64,
    condition_inverse_maxiter=20,
    condition_inner_cg_maxiter=512,
    condition_inner_cg_tol=1e-5,
    condition_num_starts=2,
    condition_agreement_tol=0.05,
    condition_eigen_residual_tol=0.05,
    condition_shifts='1e-2,1e-4',
    condition_trace_probes=4,
)

def microbatch_groups(batch_size, n_requested, data_shards):
    if batch_size <= 0 or data_shards <= 0 or batch_size % data_shards:
        raise ValueError('Batch must be positive and divisible by data shards')
    if n_requested <= 0:
        raise ValueError('Microbatch count must be positive')
    local_batch = batch_size // data_shards
    n_actual = min(n_requested, local_batch)
    q, r = divmod(local_batch, n_actual)
    groups = (
        (0, r, data_shards * (q + 1)),
        (r * data_shards * (q + 1), n_actual - r, data_shards * q),
    )
    return n_actual, tuple(group for group in groups if group[1] > 0)


def weighted_microbatch_sum(batch, groups, contribution_fn, zero):
    batch_size = batch['input_tokens'].shape[0]
    total = zero
    for offset, count, mb_size in groups:
        def body(i, carry):
            start = offset + i * mb_size
            mb = jax.tree.map(
                lambda x: jax.lax.dynamic_slice_in_dim(x, start, mb_size, axis=0),
                batch,
            )
            value = contribution_fn(mb)
            return jax.tree.map(
                lambda acc, x: acc + (mb_size / batch_size) * x, carry, value)
        total = jax.lax.fori_loop(0, count, body, total)
    return total

def get_gpu_memory():
    try:
        command = "nvidia-smi --query-gpu=memory.free --format=csv"
        memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
        memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
        return memory_free_values
    except Exception:
        return [0]

def is_embedding_param(param_name, param_value):
    if 'embedding' in param_name:
        return True
    return False

def count_params(params):
    non_embedding_count = 0
    total_count = 0

    for param_name, param_value in jax.tree_util.tree_leaves_with_path(params):
        # print(param_name[-1].key, is_embedding_param(param_name[-1].key, param_value), jnp.prod(jnp.array(param_value.size)))
        total_count += jnp.prod(jnp.array(param_value.size))
        if not is_embedding_param(param_name[-1].key, param_value):
            non_embedding_count += jnp.prod(jnp.array(param_value.size))
            print(param_name[-5:], is_embedding_param(param_name[-1].key, param_value), jnp.prod(jnp.array(param_value.size)))
        else:
            print(param_name, is_embedding_param(param_name[-1].key, param_value), jnp.prod(jnp.array(param_value.size)))
    # print(non_embedding_count)
    return total_count, non_embedding_count



def get_tpu_metrics():
    """Snapshot a few TPU utilization/memory metrics via the libtpu monitoring SDK.
    Returns an empty dict if unavailable (e.g. not running on TPU, or metrics
    server not yet up) so this never crashes a training run."""
    metric_names = ["duty_cycle_pct", "tensorcore_util", "hbm_capacity_usage", "hbm_capacity_total"]
    out = {}
    for name in metric_names:
        try:
            result = tpu_monitoring.get_metric(name)
            data = result.data()
            # data() returns a list of str values per chip; cast + average across chips
            values = [float(v) for v in data]
            if values:
                out[f"tpu_{name}"] = sum(values) / len(values)
        except Exception as e:
            pass
    return out


def main(argv):
    JaxDistributedConfig.initialize(FLAGS.jax_distributed)

    if FLAGS.condition_log:
        if FLAGS.optimizer_type not in ('cg', 'muon') or (
                FLAGS.optimizer_type == 'muon' and not FLAGS.gauss_newton):
            raise ValueError('Condition diagnostics support CG and Muon-GN only')
        if FLAGS.condition_every <= 0 or FLAGS.condition_top_maxiter < 3 \
                or FLAGS.condition_inverse_maxiter < 3 \
                or FLAGS.condition_inner_cg_maxiter <= 0 \
                or FLAGS.condition_num_starts < 2 \
                or FLAGS.condition_trace_probes < 2:
            raise ValueError('Condition diagnostics require positive budgets, '
                             'at least three outer iterations, and two starts')
        if not 0 < FLAGS.condition_inner_cg_tol < 1:
            raise ValueError('condition_inner_cg_tol must be in (0, 1)')
        if not 0 < FLAGS.condition_agreement_tol < 1 \
                or not 0 < FLAGS.condition_eigen_residual_tol < 1:
            raise ValueError('Condition validation tolerances must be in (0, 1)')
        shifts = [float(value) for value in FLAGS.condition_shifts.split(',')]
        if not shifts or any(not math.isfinite(value) or value <= 0
                             for value in shifts):
            raise ValueError('condition_shifts must be positive finite ratios')

    if not 0.0 <= FLAGS.outer_weight_decay < 1.0:
        raise ValueError("outer_weight_decay must satisfy 0 <= rho < 1")
    if FLAGS.outer_weight_decay and (
        FLAGS.optimizer_type != 'muon' or not FLAGS.gauss_newton
        or FLAGS.adaptive_inner_loop or FLAGS.weight_average
    ):
        raise ValueError("outer decay requires non-adaptive Muon-GN and weight_average=False")
    if FLAGS.train_batch_growth_interval < 0:
        raise ValueError("train_batch_growth_interval must be nonnegative")
    cg_resume.validate_batch_lambda(
        FLAGS.cg_lambda_batch_denominator, FLAGS.optimizer_type,
        FLAGS.cg_lambda_final, FLAGS.cg_lambda_ramp_steps, False,
        max(FLAGS.train_dataset_batch_size,
            FLAGS.train_batch_max if FLAGS.train_batch_growth_interval > 0 else 0,
            FLAGS.train_dataset.huggingface_dataset.batch_size))
    lambda_schedule_enabled = (
        FLAGS.cg_lambda_final != -1.0 or FLAGS.cg_lambda_ramp_steps != 0)
    if FLAGS.optimizer_type != 'cg' and lambda_schedule_enabled:
        raise ValueError("CG lambda scheduling is only available for optimizer_type='cg'")
    if FLAGS.optimizer_type == 'cg':
        if not 0.0 <= FLAGS.cg_interpolation_lambda <= 1.0:
            raise ValueError("cg_interpolation_lambda must be in [0, 1]")
        if FLAGS.cg_lambda_final != -1.0:
            if not 0.0 <= FLAGS.cg_lambda_final <= 1.0:
                raise ValueError("cg_lambda_final must be in [0, 1]")
            if FLAGS.cg_lambda_ramp_steps <= 0:
                raise ValueError("cg_lambda_ramp_steps must be positive when a final lambda is set")
        elif FLAGS.cg_lambda_ramp_steps != 0:
            raise ValueError("cg_lambda_ramp_steps requires cg_lambda_final")

    output_dir = os.path.join(FLAGS.output_dir, FLAGS.experiment_id)
    variant = mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = mlxu.user_flags_to_config_dict(FLAGS, FLAGS_DEF)

    cg_metadata = None
    cg_generation = -1
    if FLAGS.cg_resume_state:
        if FLAGS.optimizer_type != 'cg':
            raise ValueError('cg_resume_state requires optimizer_type=cg')
        cg_metadata, cg_dataset, cg_marker = cg_resume.read_bundle(FLAGS.cg_resume_state)
        cg_resume.validate_consumption(cg_metadata)
        cg_resume.validate_flags(cg_metadata['flags'], flags_config_dict)
        cg_generation = cg_marker['generation']
    if FLAGS.optimizer_type == 'cg' and (
            FLAGS.cg_resume_state or FLAGS.save_model_freq > 0 or FLAGS.save_milestone_freq > 0):
        if (FLAGS.train_dataset.type != 'huggingface'
                or FLAGS.train_dataset.huggingface_dataset.pretokenized_dataset_dir):
            raise ValueError('CG checkpointing requires raw HuggingFace packed data')
        if FLAGS.eval_steps > 0 and (FLAGS.eval_freq or FLAGS.log_initial_eval):
            if (FLAGS.eval_dataset.type != 'huggingface'
                    or FLAGS.eval_dataset.huggingface_dataset.pretokenized_dataset_dir):
                raise ValueError('CG checkpointing requires restartable raw HF evaluation')

    log_config = mlxu.flatten_config_dict(flags_config_dict)

    set_random_seed(FLAGS.seed)

    print(FLAGS.train_dataset)
    init_checkpoint_path = FLAGS.load_checkpoint
    if init_checkpoint_path.startswith('trainstate::'):
        raise ValueError(
            'Full-state GN continuation is not supported; start a new params-only branch')
    branch_parent_metadata = None
    branch_parent_complete = None
    is_muon_gn_packed_branch = (
        FLAGS.optimizer_type == 'muon' and FLAGS.gauss_newton
        and init_checkpoint_path.startswith('trainstate_params::')
        and bool(FLAGS.load_dataset_state)
        and FLAGS.train_dataset.type == 'huggingface'
        and not FLAGS.train_dataset.huggingface_dataset.pretokenized_dataset_dir
    )
    if is_muon_gn_packed_branch:
        parent_path = init_checkpoint_path.split('::', 1)[1]
        parent_paths = resume_companion_paths(parent_path)

        def load_parent_companion(name):
            path = parent_paths[name]
            if path.startswith('gs://'):
                path = load_from_gcs(
                    path, os.path.join(FLAGS.tmp_dir, f'branch_parent_{name}.pkl'))
            return mlxu.load_pickle(path)

        branch_parent_metadata = load_parent_companion('metadata')
        branch_parent_complete = load_parent_companion('complete')

    if not FLAGS.cg_resume_state and FLAGS.load_checkpoint.split('::')[-1].startswith('gs://'):
        FLAGS.load_checkpoint = load_ckpt_from_gcs(FLAGS.load_checkpoint, local_path=os.path.join(FLAGS.tmp_dir, 'model.ckpt'))
    if FLAGS.train_dataset.huggingface_dataset.pretokenized_dataset_dir.startswith('gs://'):
        num_to_download = FLAGS.gcs_num_train_files_to_download # Files download around 100 MiB/s
        tmp_dir = FLAGS.tmp_dir
        load_first_n_files_from_gcs(os.path.join(FLAGS.train_dataset.huggingface_dataset.pretokenized_dataset_dir, 'train'), os.path.join(tmp_dir, 'train_dataset/train'), num_to_download=num_to_download)
        modify_dataset_info_gcs(os.path.join(FLAGS.train_dataset.huggingface_dataset.pretokenized_dataset_dir, 'train/dataset_info.json'), os.path.join(tmp_dir, 'train_dataset/train'), num_files_to_keep=num_to_download)
        modify_state_json_gcs(os.path.join(FLAGS.train_dataset.huggingface_dataset.pretokenized_dataset_dir, 'train/state.json'), os.path.join(tmp_dir, 'train_dataset/train'), num_files_to_keep=num_to_download)
        load_from_gcs(os.path.join(FLAGS.train_dataset.huggingface_dataset.pretokenized_dataset_dir, 'dataset_dict.json'), os.path.join(tmp_dir, 'train_dataset/dataset_dict.json'))
        FLAGS.train_dataset.huggingface_dataset.pretokenized_dataset_dir = os.path.join(tmp_dir, 'train_dataset')
    if FLAGS.eval_dataset.huggingface_dataset.pretokenized_dataset_dir.startswith('gs://'):
        FLAGS.eval_dataset.huggingface_dataset.pretokenized_dataset_dir = load_from_gcs(FLAGS.eval_dataset.huggingface_dataset.pretokenized_dataset_dir, os.path.join(FLAGS.tmp_dir,'eval_dataset'))
    if not FLAGS.cg_resume_state and FLAGS.load_dataset_state.startswith('gs://'):
        FLAGS.load_dataset_state = load_from_gcs(
            FLAGS.load_dataset_state,
            os.path.join(FLAGS.tmp_dir, 'dataset_state.pkl'),
        )

    tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer)
    dataset = DatasetFactory.load_dataset(FLAGS.train_dataset, tokenizer)
    if FLAGS.load_dataset_state or FLAGS.cg_resume_state:
        dataset_state = cg_dataset if cg_metadata is not None else mlxu.load_pickle(FLAGS.load_dataset_state)
        if dataset_state is None:
            raise ValueError('Checkpoint has no dataset state')
        if isinstance(dataset, HuggingfaceDataset):
            # A fresh Muon-GN branch can regroup the saved token stream.
            # Other initialization routes keep strict batch-size validation.
            if is_muon_gn_packed_branch and dataset_state.get('packed_state_version') != 1:
                raise ValueError('Muon-GN branching requires a packed dataset checkpoint')
            if is_muon_gn_packed_branch:
                validate_branch_parent(
                    branch_parent_metadata, branch_parent_complete, dataset_state,
                    FLAGS.log_step_offset, FLAGS.log_token_offset)
            dataset.load_state_dict(
                dataset_state,
                allow_batch_size_change=is_muon_gn_packed_branch,
            )
        else:
            dataset.load_state_dict(dataset_state)
        print('loaded dataset state', flush=True)

    if FLAGS.eval_steps > 0:
        eval_dataset = DatasetFactory.load_dataset(
            FLAGS.eval_dataset, dataset.tokenizer
        )
        eval_iterator = iter(eval_dataset)

    seq_length = dataset.seq_length
    llama_config = LLaMAConfigurator.finalize_config(FLAGS.llama)
    if FLAGS.condition_log and FLAGS.optimizer_type == 'muon':
        stochastic = ('embedding_dropout', 'feedforward_dropout',
                      'attention_dropout', 'residue_dropout', 'fcm_min_ratio',
                      'fcm_max_ratio')
        enabled = [name for name in stochastic
                   if float(getattr(llama_config, name, 0.0)) != 0.0]
        if enabled:
            raise ValueError('Muon condition diagnostics require dropout/FCM '
                             'disabled: ' + ', '.join(enabled))

    model = FlaxLLaMAForCausalLMModule(
        llama_config,
        dtype=get_float_dtype_by_name(FLAGS.dtype),
        param_dtype=get_float_dtype_by_name(FLAGS.param_dtype),
    )

    def get_global_lr_sched(method, lr, taylor_steps, inner_loop_iter, warmup, inner_warmup, end_lr):
        if method == 'global_cosine':
            decay_steps = taylor_steps*inner_loop_iter
            decay_steps = int(decay_steps)
            if warmup <= 1.0:
                warmup = int(warmup*decay_steps)

            if isinstance(warmup, tuple):
                warmup = int(warmup[0])
            
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=lr*0.1,
                peak_value=lr,
                warmup_steps=warmup,
                decay_steps=decay_steps,
                end_value=end_lr,
            )
        elif method == 'cosine_with_global_schedule':
            decay_steps = taylor_steps
            decay_steps = int(decay_steps)
            if warmup <= 1.0:
                warmup = int(warmup*decay_steps)
            if isinstance(warmup, tuple):
                warmup = int(warmup[0])

            if inner_warmup <= 1.0:
                inner_warmup = int(inner_warmup*inner_loop_iter)
            if isinstance(inner_warmup, tuple):
                inner_warmup = int(inner_warmup[0])
            
            global_sched = optax.warmup_cosine_decay_schedule(
                init_value=lr*0.1,
                peak_value=lr,
                warmup_steps=warmup,
                decay_steps=decay_steps,
                end_value=end_lr,
            )
            schedules = []
            boundaries = []
            for step in range(taylor_steps):
                peak_lr = global_sched(step)
                inner_sched = optax.warmup_cosine_decay_schedule(
                    init_value=peak_lr*0.1,
                    peak_value=peak_lr,
                    warmup_steps=inner_warmup,
                    decay_steps=inner_loop_iter,
                    end_value=end_lr,
                )
                schedules.append(inner_sched)
                boundaries.append(step*inner_loop_iter)

            schedule = optax.join_schedules(schedules, boundaries)

        elif method == 'constant_with_inner_cosine':
            decay_steps = taylor_steps
            decay_steps = int(decay_steps)
            if warmup <= 1.0:
                warmup = int(warmup*decay_steps)
            if isinstance(warmup, tuple):
                warmup = int(warmup[0])

            if inner_warmup <= 1.0:
                inner_warmup = int(inner_warmup*inner_loop_iter)
            if isinstance(inner_warmup, tuple):
                inner_warmup = int(inner_warmup[0])

            if warmup == 0:
                init_value = lr
            else:
                init_value = lr*0.1
            
            global_sched = optax.warmup_constant_schedule(
                init_value=init_value,
                peak_value=lr,
                warmup_steps=warmup,
            )
            schedules = []
            boundaries = []
            for step in range(taylor_steps):
                peak_lr = global_sched(step)
                inner_sched = optax.warmup_cosine_decay_schedule(
                    init_value=peak_lr*0.1,
                    peak_value=peak_lr,
                    warmup_steps=inner_warmup,
                    decay_steps=inner_loop_iter,
                    end_value=end_lr,
                )
                schedules.append(inner_sched)
                boundaries.append((step+1)*inner_loop_iter)

            schedule = optax.join_schedules(schedules, boundaries[:-1])

        elif method == 'constant':
            schedule = optax.constant_schedule(lr)
        else:
            raise ValueError(f"Unknown global schedule method: {method}")

        return schedule

    def build_optimizer(lr_sched, b1, b2, grad_clip=None, wd=0.0, optimizer_type='adamw'):
        if optimizer_type == 'adamw':
            if grad_clip:
                optimizer = optax.chain(
                    optax.clip_by_global_norm(grad_clip),
                    optax.adamw(
                        learning_rate=lr_sched,
                        b1=b1,
                        b2=b2,
                        mu_dtype=jnp.float32,
                        weight_decay=wd
                    )
                )
            else:
                optimizer = optax.adamw(
                    learning_rate=lr_sched,
                    b1=b1,
                    b2=b2,
                    mu_dtype=jnp.float32,
                    weight_decay=wd
                )
        elif optimizer_type == 'muon':
            adamw_chain = optax.chain(
                optax.clip_by_global_norm(grad_clip),
                optax.adamw(
                    learning_rate=lr_sched,
                    weight_decay=wd,
                    b1=b1,
                    b2=b2,
                    mu_dtype=jnp.float32,
                ),
            )

            muon_chain = optax.chain(
                optax.clip_by_global_norm(grad_clip),
                optax.contrib.muon(
                    learning_rate=lr_sched,
                    adam_weight_decay=wd,
                    adam_b1=b1,
                    adam_b2=b2,
                    mu_dtype=jnp.float32,
                ),
            )

            transform_dict = {
                'adamw': adamw_chain,
                'muon':   muon_chain,
            }

            def create_param_selector(params):
                """
                Return a pytree (same structure as params) whose leaves are strings
                ('adamw' or 'muon'), AND print out the name of each parameter and its assignment.
                """
                # 1) Flatten the nested param dict so we get name tuples -> arrays
                flat_params = flatten_dict(params, sep='.')

                # Define first and last layer parameter names
                first_layer_keys = ['params.transformer.wte.embedding']
                last_layer_keys = ['params.lm_head.kernel']

                # 2) Build the selector tree
                flat_selector = {}
                for name_tuple, param in flat_params.items():
                    # print(name_tuple)
                    if name_tuple in first_layer_keys or name_tuple in last_layer_keys:
                        # print(f"Assigning param '{name_tuple}' (shape={param.shape}) to ADAMW.")
                        flat_selector[name_tuple] = 'adamw'
                    else:
                        # print(f"Assigning param '{name_tuple}' (shape={param.shape}) to MUON.")
                        flat_selector[name_tuple] = 'muon'

                # 3) Unflatten back to the original param-tree structure
                selector_tree = unflatten_dict(flat_selector, sep='.')
                return selector_tree

        
            def param_selector(params):
                return create_param_selector(params)
            
            optimizer = optax.multi_transform(transform_dict, param_selector)
        elif optimizer_type == 'cg':
            # CG doesn't use an optax optimizer at all -- inert placeholder
            # so tayl_solver.init(...) still produces a validly-shaped opt_state.
            optimizer = optax.set_to_zero()
        return optimizer

    _, optimizer_info = OptimizerFactory.get_optimizer(FLAGS.optimizer)
    # Use the exact same learning-rate schedule as the regular AdamW path.
    adamw_lr_schedule = optimizer_info['learning_rate_schedule']
    lr_sched = get_global_lr_sched(FLAGS.lr_sched, FLAGS.inner_loop_lr, FLAGS.total_steps, FLAGS.inner_loop_iter, FLAGS.global_warmup, FLAGS.inner_loop_warmup, FLAGS.end_lr)
    tayl_solver = build_optimizer(lr_sched, FLAGS.inner_b1, FLAGS.inner_b2, FLAGS.inner_clip_gradient, FLAGS.optimizer_wd, FLAGS.optimizer_type)

    # optimizer, optimizer_info = OptimizerFactory.get_optimizer(FLAGS.optimizer)

    def create_trainstate_from_params(params):
        return CustomTrainState.create(params=params, tx=tayl_solver, apply_fn=None)

    def init_fn(rng):
        rng_generator = JaxRNG(rng)
        params = model.init(
            input_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            position_ids=jnp.zeros((4, seq_length), dtype=jnp.int32),
            attention_mask=jnp.ones((4, seq_length), dtype=jnp.int32),
            rngs=rng_generator(LLaMAConfigurator.rng_keys()),
        )
        return CustomTrainState.create(params=params, tx=tayl_solver, apply_fn=None)

    def train_step_jvp(train_state, params0, rng, batch, wd):
        rng_generator = JaxRNG(rng)

        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))

        def loss_and_accuracy(params0, params):
            dparams = jax.tree_util.tree_map(lambda x, y: x - y, params, params0)
            def f_batch(p):
                logits = model.apply(
                    p, batch['input_tokens'], deterministic=False,
                    rngs=rng_generator(LLaMAConfigurator.rng_keys()),
                ).logits
                return logits
            primals, Jx = jax.jvp(f_batch, (params0,), (dparams,))
            logits = primals + jax.lax.stop_gradient(Jx)
            return cross_entropy_loss_and_accuracy_with_weight_decay(
                logits, batch['target_tokens'], train_state.params, params0, batch['loss_masks'], weight_decay=wd
            )
        grad_fn = jax.value_and_grad(loss_and_accuracy, has_aux=True)
        (loss, accuracy), grads = grad_fn(params0, train_state.params)
        try:
            perplexity = jnp.exp(loss)
        except OverflowError:
            perplexity = jnp.float32("inf")
        train_state = train_state.apply_gradients(grads=grads)
        metrics = dict(
            loss=loss,
            perplexity=perplexity,
            accuracy=accuracy,
            learning_rate=lr_sched(train_state.step),
            gradient_norm=global_norm(grads),
            param_norm=global_norm(train_state.params),
            gpu_memory=get_gpu_memory()[0],
        )
        return train_state, rng_generator(), metrics


    def train_step_gauss_newton(train_state, params0, rng, batch, wd, is_last_step):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))

        def f_batch(p):
            out = model.apply(
                p,
                batch['input_tokens'],
                deterministic=False,              
                rngs=rng_generator(LLaMAConfigurator.rng_keys()),
            )
            return out.logits                    # [B, ..., vocab]

        def scalar_loss_on_logits(logits):
            loss, _ = cross_entropy_loss_and_accuracy_with_weight_decay(
                logits, batch['target_tokens'], train_state.params, params0, batch['loss_masks'], weight_decay=wd
            )
            return loss

        def value_and_gradient(params0, params, is_last_step):
            '''
            ∇θ [ L(y0) + g0·v + 1/2 v^T G0 v ]
            = g0 + H0 v
                g0 = ∂L/∂p at p0 = ∂L/∂f @ ∂f/∂p at p0 ;  
            H0 v = (∂²L/∂p² at p0) @ v = (g0^T ∂²L/∂f² g0) (dθ) = (J(p0)^T ∂²L/∂f² J(p0) (dθ))
            '''
            # Linearize f at params0
            logits0, jvp_fn = linearize(f_batch, params0)          # y0,   v = J(p0) dθ

            # dθ and forward-mode JVP: v = J0 (params - params0)
            dparams = jax.tree_util.tree_map(lambda x, y: x - y, params, params0)
            v = jvp_fn(dparams)

            # g0 = ∂L/∂y at y0 ;  Hv = (∂²L/∂y² at y0) @ v
            grad_Ly = jax.grad(scalar_loss_on_logits)              # y -> grad wrt logits
            g0 = grad_Ly(logits0) # ∂L/∂f at p0
            _, Hv = jax.jvp(grad_Ly, (logits0,), (v,))            # Hessian-vector (logits space) = (∂²L/∂f² at p0) J(p0) dθ

            # Single pullback: J0^T (g0 + H0 v)
            jt_fn = linear_transpose(jvp_fn, params0) # primals just for shape/dtype
            (grad_params,) = jt_fn(jax.tree_util.tree_map(lambda a, b: a + b, g0, Hv))
            b_norm = jax.lax.cond(
                is_last_step,
                lambda: global_norm(jt_fn(g0)[0]),
                lambda: jnp.float32(0.0),
            )

            # quadratic loss on linear model
            loss = scalar_loss_on_logits(logits0) + jnp.sum(g0 * v) + 0.5 * jnp.sum(v * Hv)


            return (loss, 0), (grad_params, b_norm)

        (loss, accuracy), (grads, b_norm) = value_and_gradient(params0, train_state.params, is_last_step)

        try:
            perplexity = jnp.exp(loss)
        except OverflowError:
            perplexity = jnp.float32("inf")

        train_state = train_state.apply_gradients(grads=grads)

        metrics = dict(
            linear_model_loss=loss,
            perplexity=perplexity,
            accuracy=accuracy,
            learning_rate=lr_sched(train_state.step),
            gradient_norm=global_norm(grads),
            b_norm=b_norm,
            relative_residual=global_norm(grads) / (b_norm + 1e-12),
            param_norm=global_norm(train_state.params),
            gpu_memory=get_gpu_memory()[0],
        )
        return train_state, rng_generator(), metrics


    def loss_fn(params, batch, rng):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        logits = model.apply(
            params, batch['input_tokens'], deterministic=False,
            rngs=rng_generator(LLaMAConfigurator.rng_keys()),
        ).logits
        return cross_entropy_loss_and_accuracy(
            logits, batch['target_tokens'], batch['loss_masks']
        )


    def eval_step(params, rng, batch):
        rng_generator = JaxRNG(rng)
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        logits = model.apply(
            params, batch['input_tokens'], deterministic=True,
            rngs=rng_generator(LLaMAConfigurator.rng_keys()),
        ).logits
        loss, accuracy = cross_entropy_loss_and_accuracy(
            logits, batch['target_tokens'], batch['loss_masks']
        )
        try:
            perplexity = jnp.exp(loss)
        except OverflowError:
            perplexity = jnp.float32("inf")
        metrics = dict(
            eval_loss=loss,
            eval_accuracy=accuracy,
            eval_perplexity=perplexity,
        )
        return rng_generator(), metrics
        

    def train_step_cg(
        params0,          # base parameters θ_0 for this outer step
        first_moment,     # persistent Adam first moment m_{t-1}
        second_moment,    # persistent Adam second moment s_{t-1}
        cg_x0,            # warm-start for CG (previous step's solution y, in preconditioned space)
        adam_step,        # outer step count, used for bias correction
        outer_step,       # same counter, used for the LR schedule
        rng,
        batch,
        wd,
        scheduled_lambda,
    ):
        rng_generator = JaxRNG(rng)
        batch_ = with_sharding_constraint(batch, PS(('dp', 'fsdp')))

        # Partition by per-device rows. Unequal groups have distinct static slice
        # sizes, while every global slice remains evenly data-sharded.
        batch_size = batch_['input_tokens'].shape[0]
        _, groups = microbatch_groups(batch_size, FLAGS.cg_n_micro, data_shards)

        def gradient_contribution(mb):
            def f_mb(p):
                return model.apply(p, mb['input_tokens'], deterministic=True).logits

            def scalar_loss_mb(logits):
                return cross_entropy_loss_and_accuracy_with_weight_decay(
                    logits, mb['target_tokens'], params0, params0,
                    mb['loss_masks'], weight_decay=wd)[0]

            logits0_mb, jvp_fn_mb = linearize(f_mb, params0)
            g0_mb = jax.grad(scalar_loss_mb)(logits0_mb)
            return linear_transpose(jvp_fn_mb, params0)(g0_mb)[0]

        b_param = weighted_microbatch_sum(
            batch_, groups, gradient_contribution,
            jax.tree.map(jnp.zeros_like, params0))

        # ── Adam EMA updates ──────────────────────────────────────────
        #
        # Both moments are updated once per outer step  using the full-batch gradient b_param
        # The bias-corrected moments are used to construct the CG right-hand side and the Adam diagonal preconditioner.
        new_adam_step = adam_step + 1
        adam_lr = adamw_lr_schedule(outer_step)

        # First moment: exponential moving average of b_param (the gradient).
        new_first_moment = jax.tree_util.tree_map(
            lambda m, g: FLAGS.optimizer.adamw_optimizer.b1 * m
            + (1.0 - FLAGS.optimizer.adamw_optimizer.b1) * g,
            first_moment,
            b_param,
        )

        # Second moment: exponential moving average of b_param^2 (the gradient variance).
        new_second_moment = jax.tree_util.tree_map(
            lambda s, g: FLAGS.optimizer.adamw_optimizer.b2 * s
            + (1.0 - FLAGS.optimizer.adamw_optimizer.b2) * jnp.square(g),
            second_moment,
            b_param,
        )

        # Bias corrections for Adam: account for the zero-initialization of moments.
        beta1_correction = 1.0 - jnp.power(
            jnp.asarray(FLAGS.optimizer.adamw_optimizer.b1, dtype=jnp.float32),
            new_adam_step,
        )
        beta2_correction = 1.0 - jnp.power(
            jnp.asarray(FLAGS.optimizer.adamw_optimizer.b2, dtype=jnp.float32),
            new_adam_step,
        )

        # ── build the CG right-hand side ─────────────────────────────
        # Build the Adam-based CG RHS: -m_hat_t.
        rhs = jax.tree_util.tree_map(
            lambda m: -m / beta1_correction,
            new_first_moment,
        )

        
        adam_eps = jnp.asarray(1e-8, dtype=jnp.float32)

        interpolation_lambda = jnp.asarray(scheduled_lambda, dtype=jnp.float32)

        # Protect against division by zero at the first warmup step.
        safe_adam_lr = jnp.maximum(
            adam_lr,
            jnp.asarray(1e-12, dtype=jnp.float32),
        )

        # ── Adam Interpolation ────────────────────────────
        #
        # The interpolated operator is A_t(v) = λG + (1-λ)/η * D_t, where:
        #   G   = J^T H J   (Gauss-Newton curvature)
        #   D_t = diag(sqrt(s_hat_t) + eps)   (Adam second-moment diagonal)
        def apply_D_inv(tree):
            """
            Apply the inverse Adam diagonal D_t^{-1} elementwise:
        
                D_t = sqrt(s_hat_t) + eps
        
                D_t^{-1} v = v / (sqrt(s_hat_t) + eps).
        
            The diagonal is never materialized; the operation is applied lazily
            to the parameter pytree.
            """
            return jax.tree_util.tree_map(
                lambda value, second_moment: (
                    value / (
                        jnp.sqrt(second_moment / beta2_correction)
                        + adam_eps
                    )
                ),
                tree,
                new_second_moment,
            )

        def apply_G(v):
            """Apply the weighted full-batch Gauss--Newton operator."""
            def curvature_contribution(mb):
                def f_mb(p):
                    return model.apply(p, mb['input_tokens'], deterministic=True).logits

                def scalar_loss_mb(logits):
                    return cross_entropy_loss_and_accuracy_with_weight_decay(
                        logits, mb['target_tokens'], params0, params0,
                        mb['loss_masks'], weight_decay=wd)[0]

                logits0_mb, jvp_fn_mb = linearize(f_mb, params0)
                grad_loss = jax.grad(scalar_loss_mb)
                jt_fn_mb = linear_transpose(jvp_fn_mb, params0)
                logits_v_mb = jvp_fn_mb(v)
                _, hv_mb = jax.jvp(grad_loss, (logits0_mb,), (logits_v_mb,))
                return jt_fn_mb(hv_mb)[0]

            return weighted_microbatch_sum(
                batch_, groups, curvature_contribution,
                jax.tree.map(jnp.zeros_like, params0))

        # ── CG operator Av ────────────────────────────────
        # Av(v) computes A_t(v) = λ G v + (1-λ)/η D_t v.
        # CG calls this repeatedly to solve A_t x = rhs.
        def Av(v):
            Gv_param = apply_G(v)
            # Add the Adam diagonal contribution: (1-λ)/η * I
            return jax.tree_util.tree_map(
                lambda gv, vi, second_moment: (
                    interpolation_lambda * gv
                    + (
                        (1.0 - interpolation_lambda)
                        / safe_adam_lr
                    )
                    * (
                        jnp.sqrt(second_moment / beta2_correction)
                        + adam_eps
                    )
                    * vi
                ),
                Gv_param,
                v,
                new_second_moment,
            )

        # ── Run preconditioned CG ─────────────────────────────────────
        #
        # Solve the original interpolated system:
        #
        #     A_t x = rhs
        #
        # where
        #
        #     A_t = λ G + (1-λ)/η * D_t.
        #
        # JAX CG uses D_t^{-1} as the preconditioner M ≈ A_t^{-1}.
        x, _ = cg(
            Av,
            rhs,
            x0=cg_x0,
            tol=FLAGS.cg_tol,
            atol=FLAGS.cg_atol,
            maxiter=FLAGS.cg_maxiter,
            M=apply_D_inv,
        )

        # Compute residual for logging 
        # relative_residual = ||A x - rhs|| / ||rhs||
        residual = jax.tree_util.tree_map(
            lambda ax, rhs_leaf: ax - rhs_leaf,
            Av(x),
            rhs,
        )
        
        residual_norm = global_norm(residual)
        rhs_norm = global_norm(rhs)
        relative_residual = residual_norm / (rhs_norm + 1e-12)        

        # ── Apply update with decoupled weight decay ──────────────────
        #
        # Full parameter update:
        #   θ_new = θ_0 + x - adam_fraction * η * λ_wd * θ_0
        # where adam_fraction = (1 - λ) scales the AdamW weight decay by how
        # much of the operator is the Adam diagonal (vs the GN term).
        adam_fraction = 1.0 - interpolation_lambda
        weight_decay = jnp.asarray(
            FLAGS.optimizer.adamw_optimizer.weight_decay,
            dtype=jnp.float32,
        )
        # Fused: avoids materializing an intermediate parameter pytree for p + update.
        new_params = jax.tree_util.tree_map(
            lambda p, update: p
            + update
            - adam_fraction * adam_lr * weight_decay * p,
            params0,
            x,
        )

        # ── Collect metrics ──────────────────────────────────────────
        metrics = {
            'linear_model_loss': jnp.float32(0.0),
            'gradient_norm': residual_norm,
            'param_norm': global_norm(new_params),
            'gpu_memory': get_gpu_memory()[0],
            'learning_rate': adam_lr,
            'adamw_learning_rate': adam_lr,
            'adamw_weight_decay': weight_decay,
            'b_norm': rhs_norm,
            'relative_residual': relative_residual,
            'accuracy': jnp.int32(0),
            'perplexity': jnp.float32(0.0),
            'adam_step': new_adam_step,
            'cg_lambda_scheduled': scheduled_lambda,
            'cg_lambda_effective': interpolation_lambda,
            'cg_relative_damping': jnp.where(
                interpolation_lambda > 0.0,
                (1.0 - interpolation_lambda) /
                (safe_adam_lr * interpolation_lambda),
                jnp.asarray(jnp.nan, dtype=jnp.float32)),
        }

        return (
            new_params,
            new_first_moment,
            new_second_moment,
            x,              # returned for warm-starting next outer step's CG
            new_adam_step,
            rng_generator(),
            metrics,
        )

    def condition_apply_g(params0, batch, vector, wd):
        """Deterministic full-parameter Gv on the frozen diagnostic batch.

        For Muon with multiple inner batches this intentionally describes the
        first solve batch only. Weight decay is constant with respect to logits
        and therefore is not part of this Gauss--Newton operator.
        """
        batch = with_sharding_constraint(batch, PS(('dp', 'fsdp')))
        _, groups = microbatch_groups(
            batch['input_tokens'].shape[0], FLAGS.cg_n_micro, data_shards)

        def contribution(mb):
            def logits_fn(params):
                return model.apply(
                    params, mb['input_tokens'], deterministic=True).logits

            def logits_loss(logits):
                return cross_entropy_loss_and_accuracy_with_weight_decay(
                    logits, mb['target_tokens'], params0, params0,
                    mb['loss_masks'], weight_decay=wd)[0]

            logits0, jvp_fn = linearize(logits_fn, params0)
            grad_logits = jax.grad(logits_loss)
            _, h_jv = jax.jvp(grad_logits, (logits0,), (jvp_fn(vector),))
            return linear_transpose(jvp_fn, params0)(h_jv)[0]

        return weighted_microbatch_sum(
            batch, groups, contribution, jax.tree.map(jnp.zeros_like, params0))

    train_state_shapes = jax.eval_shape(init_fn, next_rng())
    train_state_partition = match_partition_rules(
        LLaMAConfigurator.get_partition_rules(), train_state_shapes
    )

    batch_partition = {
        'input_tokens': PS(('dp', 'fsdp')), 
        'loss_masks': PS(('dp', 'fsdp')),
        'target_tokens': PS(('dp', 'fsdp')),
    }

    shard_fns, gather_fns = make_shard_and_gather_fns(
        train_state_partition, train_state_shapes
    )
    if FLAGS.optimizer_type == 'cg':
        cg_param_shards, cg_param_gathers = make_shard_and_gather_fns(train_state_partition.params)

    checkpointer = StreamingCheckpointer(
        FLAGS.checkpointer, output_dir,
        enable=jax.process_index() == 0,
    )

    sharded_init_fn = pjit(
        init_fn,
        in_shardings=PS(),
        out_shardings=train_state_partition
    )

    sharded_create_trainstate_from_params = pjit(
        create_trainstate_from_params,
        in_shardings=(train_state_partition.params, ),
        out_shardings=train_state_partition,
        donate_argnums=(0, ),
    )

    if FLAGS.gauss_newton and FLAGS.optimizer_type != 'cg':
        sharded_train_step = pjit(
            train_step_gauss_newton,
            in_shardings=(train_state_partition, train_state_partition.params, PS(), batch_partition, PS(), PS()),
            out_shardings=(train_state_partition, PS(), PS()),
            # donate_argnums=(0, 1),
        )
    elif not FLAGS.gauss_newton:

        sharded_train_step = pjit(
            train_step_jvp,
            in_shardings=(train_state_partition, train_state_partition.params, PS(), batch_partition, PS()),
            out_shardings=(train_state_partition, PS(), PS()),
            # donate_argnums=(0, 1),
        )

    if FLAGS.optimizer_type == 'cg':
        sharded_train_step_cg = pjit(
            train_step_cg,
            in_shardings=(
                train_state_partition.params,  # params0
                train_state_partition.params,  # first_moment
                train_state_partition.params,  # second_moment
                train_state_partition.params,  # cg_x0   
                PS(),                          # adam_step
                PS(),                          # outer_step
                PS(),                          # rng
                batch_partition,               # batch
                PS(),                          # wd
                PS(),                          # scheduled_lambda
            ),
            
            out_shardings=(
                train_state_partition.params,  # new_params
                train_state_partition.params,  # new_first_moment
                train_state_partition.params,  # new_second_moment
                train_state_partition.params,  # new_cg_x0 
                PS(),                          # new_adam_step
                PS(),                          # new_rng
                PS(),                          # metrics
            ),
            donate_argnums=(1, 2, 3),
        )
    sharded_condition_apply_g = pjit(
        condition_apply_g,
        in_shardings=(train_state_partition.params, batch_partition,
                      train_state_partition.params, PS()),
        out_shardings=train_state_partition.params,
    )
    sharded_eval_step = pjit(
        eval_step,
        in_shardings=(train_state_partition.params, PS(), PS()),
        out_shardings=(PS(), PS()),
        donate_argnums=(1,),
    )

    parallel_loss_fn = jax.jit(loss_fn)

    def microbatched_loss_fn(params, batch, rng, n_micro):
        """Evaluate loss by averaging over n_micro microbatches.
        Uses a plain Python loop -- runs outside any JAX trace so no
        fori_loop is needed. Each microbatch produces its own mean-normalized
        loss; averaging recovers the full-batch loss for equal-sized splits.
        Keeps per-evaluation peak tensor size proportional to mb_size,
        not the full batch -- same principle as CG microbatching."""
        batch_size = batch['input_tokens'].shape[0]
        _, groups = microbatch_groups(batch_size, n_micro, data_shards)
        if n_micro == 1:
            loss, acc = parallel_loss_fn(params, batch, rng)
            return float(jax.device_get(loss)), float(jax.device_get(acc))
        total_loss = 0.0
        total_acc  = 0.0
        rng_key = rng
        for offset, count, mb_size in groups:
            for i in range(count):
                rng_key, subrng = jax.random.split(rng_key)
                start = offset + i * mb_size
                mb = jax.tree.map(lambda x: x[start:start + mb_size], batch)
                loss, acc = parallel_loss_fn(params, mb, subrng)
                weight = mb_size / batch_size
                total_loss += weight * float(jax.device_get(loss))
                total_acc += weight * float(jax.device_get(acc))
        return total_loss, total_acc

    def save_checkpoint(train_state, ema=None, milestone=False):
        nonlocal cg_generation
        step = int(jax.device_get(train_state.step))
        metadata = dict(
            step=step,
            variant=variant,
            flags=flags_config_dict,
            llama_config=llama_config.to_dict(),
            training_progress=progress.state_dict(),
            data_consumption_version=cg_resume.DATA_CONSUMPTION_VERSION,
        )
        if FLAGS.optimizer_type == 'cg':
            state = dict(params=train_state.params, step=train_state.step,
                         cg_first_moment=cg_first_moment, cg_second_moment=cg_second_moment,
                         cg_x0=cg_x0, cg_adam_step=cg_adam_step, sharded_rng=sharded_rng)
            def gather_replicated(x):
                return np.asarray(jax.device_get(x))

            state_gathers = dict(params=cg_param_gathers, step=gather_fns.step,
                cg_first_moment=cg_param_gathers, cg_second_moment=cg_param_gathers,
                cg_x0=cg_param_gathers, cg_adam_step=gather_replicated,
                sharded_rng=gather_replicated)
            if FLAGS.outer_momentum_beta > 0.0:
                state['outer_prev_update'] = outer_prev_update
                state_gathers['outer_prev_update'] = cg_param_gathers
            if FLAGS.weight_average:
                state['ema'] = ema
                state_gathers['ema'] = cg_param_gathers
            metadata['state_dtypes'] = {key: str(value.dtype)
                for key, value in flatten_dict(state).items()}
            metadata['process_timing'] = dict(timing.metrics(),
                                              _pending_update_s=timing._pending_update_s)
            cg_generation = cg_resume.save_bundle(
                output_dir, cg_generation, metadata, dataset_object.get_state_dict(),
                lambda path: StreamingCheckpointer.save_train_state_to_file(
                    state, path, state_gathers, float_dtype='fp32'),
                enable=checkpointer.enable)
            if milestone:
                cg_resume.retain_milestone(
                    output_dir, cg_generation, step, enable=checkpointer.enable)
            multihost_utils.sync_global_devices(f'cg_checkpoint_complete_{step}')
            return
        checkpointer.save_all(
            train_state=train_state,
            gather_fns=gather_fns,
            metadata=metadata,
            ema=ema,
            # dataset=dataset.get_state_dict(),
            milestone=milestone,
        )
    

    def shard_batch(batch, num_devices):
        # Shard each tensor along the first axis
        sharded = {k: np.array_split(v, num_devices) for k, v in batch.items()}
        # Group the shards for each device into a list of dictionaries
        return [{k: sharded[k][i] for k in batch} for i in range(num_devices)]

    


    mesh = LLaMAConfigurator.get_jax_mesh(FLAGS.mesh_dim)
    data_shards = int(mesh.shape['dp'] * mesh.shape['fsdp'])
    if FLAGS.train_batch_growth_interval > 0:
        if not isinstance(dataset, HuggingfaceDataset):
            raise ValueError("Batch growth requires the raw HuggingfaceDataset loader")
        start = FLAGS.train_dataset_batch_size
        if dataset.config.batch_size != start:
            raise ValueError("Loader and launch initial batch sizes must match")
        if start <= 0 or FLAGS.train_batch_growth_increment <= 0:
            raise ValueError("Initial batch size and growth increment must be positive")
        if FLAGS.train_batch_max < start:
            raise ValueError("train_batch_max must be at least the initial batch size")
        if any(value % data_shards for value in (
                start, FLAGS.train_batch_growth_increment, FLAGS.train_batch_max)):
            raise ValueError("Batch start, increment, and cap must be divisible by data shards")
    print(f"Mesh axes names: {mesh.axis_names}")
    print(f"Mesh shape: {mesh.shape}")

    with mesh:
        print(mesh)
        train_state, restored_params = None, None
        warmstart_params = None
        if FLAGS.load_checkpoint != '' and not FLAGS.cg_resume_state:
            train_state, restored_params = checkpointer.load_trainstate_checkpoint(
                FLAGS.load_checkpoint, train_state_shapes, shard_fns
            )
            # distinguish between loading from train_state and loading from params
            if train_state is not None and output_dir in init_checkpoint_path: # need to distinguish between loading adam initial ckpt and taylor mid-run ckpt
                # dataset_path = os.path.join(output_dir, 'dataset.pkl')
                # dataset.load_state_dict(mlxu.load_pickle(dataset_path))
                
                if FLAGS.weight_average:
                    _, ema = checkpointer.load_trainstate_checkpoint(
                        FLAGS.load_ema_checkpoint, train_state_shapes, shard_fns
                    )

                if FLAGS.train_dataset.huggingface_dataset.pretokenized_dataset_dir != '':
                    start_step = int(jax.device_get(train_state.step))
                    start_tokens = int(jax.device_get(train_state.step)) * FLAGS.train_dataset_batch_size * seq_length + FLAGS.train_dataset.huggingface_dataset.tokens_count_at_start
                    dataset.set_start_tokens(start_tokens)
                    print('loaded checkpoint, starting at step', start_step, flush=True)
                    print('\tstart tokens:', start_tokens)

            if train_state is not None: # do this in both cases
                opt_state = train_state.opt_state
                if train_state.warmstart_params:
                    warmstart_params = train_state.warmstart_params

        if train_state is None and restored_params is None:
            # Initialize from scratch
            train_state = sharded_init_fn(next_rng())
        elif train_state is None and restored_params is not None:
            # Restore from params but initialize train_state
            train_state = sharded_create_trainstate_from_params(restored_params)
            del restored_params

        if cg_metadata is not None:
            state_shards = dict(params=cg_param_shards, step=shard_fns.step,
                cg_first_moment=cg_param_shards, cg_second_moment=cg_param_shards,
                cg_x0=cg_param_shards, outer_prev_update=cg_param_shards,
                ema=cg_param_shards)
            cg_state_path = FLAGS.cg_resume_state
            if cg_state_path.startswith('gs://'):
                cg_state_path = load_from_gcs(cg_state_path, os.path.join(FLAGS.tmp_dir, 'cg_resume.msgpack'))
            restored_cg = checkpointer.load_checkpoint(cg_state_path, shard_fns=state_shards)
            restored_cg = unflatten_dict({key: value.astype(cg_metadata['state_dtypes'][key])
                for key, value in flatten_dict(restored_cg).items()})
            train_state = train_state.replace(params=restored_cg['params'], step=restored_cg['step'])
            restored_step = int(jax.device_get(restored_cg['step']))
            restored_adam_step = int(jax.device_get(restored_cg['cg_adam_step']))
            if restored_step != cg_metadata['step']:
                raise ValueError('CG state step does not match metadata')
            if restored_adam_step != restored_step:
                raise ValueError('CG Adam step does not match outer step')

        # param_count = sum(x.size for x in jax.tree_leaves(train_state.params))
        param_count, param_count_nonembed = count_params(train_state.params)
        param_count = jax.device_get(param_count)
        param_count_nonembed = jax.device_get(param_count_nonembed)

        flags_config_dict['param_count'] = param_count
        flags_config_dict['param_count_nonembed'] = param_count_nonembed
        # Memory breakdown diagnostic
        param_mem_gb = param_count * 4 / 1e9  # fp32 = 4 bytes
        optimizer_mem_gb = param_count * 4 * 2 / 1e9  # muon: ~2x params for momentum
        hbm_info = jax.devices()[0].memory_stats()
        total_hbm_gb = hbm_info.get("bytes_limit", 0) / 1e9
        used_hbm_gb = hbm_info.get("bytes_in_use", 0) / 1e9
        print(f"\n=== Memory Breakdown ===")
        print(f"  Parameters:          {param_mem_gb:.2f} GB ({param_count/1e6:.1f}M params @ fp32)")
        print(f"  Optimizer state est: {optimizer_mem_gb:.2f} GB")
        print(f"  Static total est:    {param_mem_gb + optimizer_mem_gb:.2f} GB")
        print(f"  HBM used at init:    {used_hbm_gb:.2f} GB / {total_hbm_gb:.2f} GB total")
        print(f"  HBM for activations: ~{total_hbm_gb - used_hbm_gb:.2f} GB remaining for activations")
        print(f"  Per-chip batch size: {FLAGS.train_dataset_batch_size // jax.device_count()}")
        print(f"========================\n")

        progress = resolve_progress(
            FLAGS.log_step_offset, FLAGS.log_token_offset,
            saved_state=(cg_metadata or {}).get('training_progress'),
            branch=cg_metadata is None and init_checkpoint_path.startswith('trainstate_params::'),
        )
        flags_config_dict['training_progress'] = progress.state_dict()

        if FLAGS.wandb_run_id:
            if FLAGS.load_checkpoint and not FLAGS.cg_resume_state:
                raise ValueError('Use a new W&B run ID when replaying a checkpoint; backward history is unsupported')
            wandb.init(entity=FLAGS.wandb_entity, project=FLAGS.wandb_project, resume="must", id=FLAGS.wandb_run_id, dir=FLAGS.wandb_dir)
        else:
            wandb.init(entity=FLAGS.wandb_entity, project=FLAGS.wandb_project, config=log_config, dir=FLAGS.wandb_dir)  # Replace with your project name
            is_gcs = output_dir.startswith("gs://")

            # If not GCS, create local directory
            if not is_gcs and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Save wandb_id.txt locally first
            local_path = os.path.join(output_dir if not is_gcs else FLAGS.tmp_dir, "wandb_id.txt")

            with open(local_path, 'w+') as f:
                f.write(wandb.run.id)  # Hacky but easier than handling in train state loader

            # If output_dir is a GCS bucket, upload the file
            if is_gcs:
                gcs_path = os.path.join(output_dir, "wandb_id.txt")
                upload_to_gcs(local_path, gcs_path)

        if cg_metadata is not None:
            print(f'Resumed CG: step={restored_step}, cg_adam_step={restored_adam_step}, '
                  f'dataset_total_tokens={progress.dataset_total_tokens}, '
                  f'wandb_run_id={wandb.run.id}', flush=True)

        configure_wandb_run(wandb.run)
        wandb.config.update({'training_progress': progress.state_dict()}, allow_val_change=True)
        wandb.config.update({
            'log_time_offset_s': FLAGS.log_time_offset_s,
            'timing_scope': 'cumulative_processes' if FLAGS.optimizer_type == 'cg' else 'current_process',
            'timing_includes_first_use_compilation': True,
        }, allow_val_change=True)
        timing = ProcessTiming(**(cg_metadata or {}).get('process_timing', {}))


        start_step = int(jax.device_get(train_state.step))
        
        def copy_array(x):
            return copy.copy(x)  # or x.copy() if x is a NumPy/JAX array

        if FLAGS.save_model_freq > 0 and FLAGS.optimizer_type != 'cg':
            if FLAGS.weight_average:
                ema = jax.tree.map(copy_array, train_state.params)
                save_checkpoint(train_state, ema=ema)
            else:
                save_checkpoint(train_state)

        sharded_rng = next_rng()

        step_counter = trange(start_step, FLAGS.total_steps, ncols=0)

        assert FLAGS.train_dataset_batch_size % data_shards == 0, \
            "Batch size must be divisible by the data-sharding mesh size."
        
        
        
        if FLAGS.weight_average:
            print('Using weight average')
            ema = jax.tree.map(copy_array, train_state.params)


        inner_state = create_trainstate_from_params(train_state.params)
        dataset_object = dataset
        dataset = iter(dataset_object)
        actual_solve_batch_size = None
        solve_batch_size = FLAGS.train_dataset_batch_size

        def pull_training_batch(role):
            nonlocal actual_solve_batch_size
            if FLAGS.train_batch_growth_interval > 0 and role == 'solve':
                previous_batch_size = dataset_object.config.batch_size
                try:
                    dataset_object.config.batch_size = solve_batch_size
                    batch, metadata = next(dataset)
                finally:
                    dataset_object.config.batch_size = previous_batch_size
            else:
                batch, metadata = next(dataset)
            if role == 'solve':
                actual_solve_batch_size = int(batch['input_tokens'].shape[0])
            progress.charge(role, batch, metadata)
            return batch, metadata

        # Calibration is operator-specific: a successful shifted solve must not
        # hide an unresolved raw G (or vice versa) on later measurement steps.
        condition_calibrated = {}
        condition_pcg_solvers = {}

        def flatten_condition(prefix, report):
            """Select logger-safe scalar records; vectors never leave the module."""
            names = ('lambda_max_est', 'lambda_max_residual', 'lambda_min_est',
                     'lambda_min_residual', 'condition_est',
                     'condition_rayleigh_lower_bound_est', 'resolved',
                     'inner_solves', 'operator_matvecs', 'seconds',
                     'calibration_attempted', 'calibration_status',
                     'calibration_agreement', 'mu',
                     'lambda_min_lower_bound')
            result = {f'condition/{prefix}/{name}': report.get(name)
                      for name in names}
            result[f'condition/{prefix}/failure_reasons'] = ','.join(
                report.get('failure_reasons', ()))
            return result

        def run_condition_diagnostics(params, solve_batch, *, cg_diagonal=None,
                                      effective_lambda=None, safe_adam_lr=None):
            """Host controller over explicitly sharded immutable Gv kernels."""
            nonlocal condition_calibrated
            started = timeit.default_timer()
            key = jax.random.fold_in(jax.random.PRNGKey(0x434f4e44), step)
            def apply_g(vector, diagnostic_params, diagnostic_batch, *unused):
                return sharded_condition_apply_g(
                    diagnostic_params, diagnostic_batch, vector,
                    FLAGS.inner_loop_wd)
            g_operator_args = (params, solve_batch)
            if 'G' not in condition_pcg_solvers:
                condition_pcg_solvers['G'] = matrix_condition.make_pcg_solver(
                    apply_g, maxiter=FLAGS.condition_inner_cg_maxiter,
                    tolerance=FLAGS.condition_inner_cg_tol)
            for relative in tuple(float(value) for value in
                                  FLAGS.condition_shifts.split(',')):
                name = f'condition_shift_{relative:.0e}'.replace('e-0', 'e-')
                def dynamic_shifted(vector, diagnostic_params,
                                    diagnostic_batch, mu):
                    return matrix_condition.tree_add(
                        apply_g(vector, diagnostic_params, diagnostic_batch),
                        vector, alpha=mu)
                if name not in condition_pcg_solvers:
                    condition_pcg_solvers[name] = matrix_condition.make_pcg_solver(
                        dynamic_shifted,
                        maxiter=FLAGS.condition_inner_cg_maxiter,
                        tolerance=FLAGS.condition_inner_cg_tol)
            options = dict(top_maxiter=FLAGS.condition_top_maxiter,
                inverse_maxiter=FLAGS.condition_inverse_maxiter,
                inner_maxiter=FLAGS.condition_inner_cg_maxiter,
                inner_tol=FLAGS.condition_inner_cg_tol,
                num_starts=FLAGS.condition_num_starts,
                agreement_tol=FLAGS.condition_agreement_tol,
                residual_tol=FLAGS.condition_eigen_residual_tol, key=key)
            g_reports = matrix_condition.gauss_newton_diagnostics(
                apply_g, params,
                shifts=tuple(float(value) for value in
                             FLAGS.condition_shifts.split(',')),
                calibrate_by_operator=condition_calibrated,
                compiled_solver_by_operator=condition_pcg_solvers,
                operator_args=g_operator_args, **options)
            metrics_out = flatten_condition('G', g_reports['G'])
            total_products = g_reports['G']['operator_matvecs']
            total_inner = g_reports['G']['inner_solves']
            if (g_reports['G']['calibration_attempted']
                    and g_reports['G']['calibration_status'] == 'passed'):
                condition_calibrated['G'] = True
            for shift_name, report in g_reports['shifted'].items():
                metrics_out.update(flatten_condition(
                    f'G_{shift_name.removeprefix("condition_")}', report))
                total_products += report['operator_matvecs']
                total_inner += report['inner_solves']
                if (report['calibration_attempted']
                        and report['calibration_status'] == 'passed'):
                    condition_calibrated[shift_name] = True

            apply_a_from_g = None
            if cg_diagonal is not None:
                c = (1.0 - effective_lambda) / safe_adam_lr
                def apply_a_from_g(gv, vector):
                    return jax.tree.map(
                        lambda g, v, d: effective_lambda * g + c * d * v,
                        gv, vector, cg_diagonal)
                def apply_a(vector, diagnostic_params, diagnostic_batch,
                            diagonal, interpolation, learning_rate):
                    coefficient = (1.0 - interpolation) / learning_rate
                    gv = apply_g(vector, diagnostic_params, diagnostic_batch)
                    return jax.tree.map(
                        lambda g, v, d: interpolation * g + coefficient * d * v,
                        gv, vector, diagonal)
                a_operator_args = (params, solve_batch, cg_diagonal,
                                   effective_lambda, safe_adam_lr)
                apply_p = matrix_condition.symmetric_diagonal_operator(
                    apply_a, lambda *args: args[2])
                def inverse_diagonal(residual, _params, _batch, diagonal,
                                     _interpolation, _learning_rate):
                    return jax.tree.map(lambda value, d: value / d,
                                        residual, diagonal)
                if 'A' not in condition_pcg_solvers:
                    condition_pcg_solvers['A'] = matrix_condition.make_pcg_solver(
                        apply_a, preconditioner=inverse_diagonal,
                        maxiter=FLAGS.condition_inner_cg_maxiter,
                        tolerance=FLAGS.condition_inner_cg_tol)
                if 'A_preconditioned' not in condition_pcg_solvers:
                    condition_pcg_solvers['A_preconditioned'] = \
                        matrix_condition.make_pcg_solver(
                            apply_p, maxiter=FLAGS.condition_inner_cg_maxiter,
                            tolerance=FLAGS.condition_inner_cg_tol)
                a_report = matrix_condition.condition_diagnostic(apply_a, params,
                    preconditioner=inverse_diagonal,
                    operator_args=a_operator_args,
                    compiled_solver=condition_pcg_solvers['A'],
                    calibrate=not condition_calibrated.get('A', False),
                    calibrated=condition_calibrated.get('A', False), **options)
                p_report = matrix_condition.condition_diagnostic(apply_p, params,
                    operator_args=a_operator_args,
                    compiled_solver=condition_pcg_solvers['A_preconditioned'],
                    calibrate=not condition_calibrated.get(
                        'A_preconditioned', False),
                    calibrated=condition_calibrated.get(
                        'A_preconditioned', False), **options)
                for operator_name, report in (
                        ('A', a_report), ('A_preconditioned', p_report)):
                    if (report['calibration_attempted']
                            and report['calibration_status'] == 'passed'):
                        condition_calibrated[operator_name] = True
                metrics_out.update(flatten_condition('A', a_report))
                metrics_out.update(flatten_condition('A_preconditioned', p_report))
                metrics_out.update({f'condition/{name}': value for name, value in
                    matrix_condition.structural_lower_bounds(
                        effective_lambda, safe_adam_lr, cg_diagonal).items()})
                total_products += a_report['operator_matvecs'] + p_report['operator_matvecs']
                total_inner += a_report['inner_solves'] + p_report['inner_solves']

            # A uses the same Gz as G and therefore adds no GN products.
            probe_samples = matrix_condition.probe_operators(
                lambda vector: apply_g(vector, *g_operator_args), params,
                num_probes=FLAGS.condition_trace_probes,
                key=jax.random.fold_in(key, 0x54524143),
                apply_a_from_g=apply_a_from_g)
            total_products += FLAGS.condition_trace_probes
            for operator_name, samples in probe_samples.items():
                top = (g_reports['G']['lambda_max_est'] if operator_name == 'G'
                       else a_report['lambda_max_est'])
                summary = matrix_condition.summarize_probe_samples(*samples, top)
                metrics_out.update({f'condition/{operator_name}/{name}': value
                                    for name, value in summary.items()})
            metrics_out['condition/operator_matvecs'] = total_products
            metrics_out['condition/inner_solves'] = total_inner
            metrics_out['condition/seconds'] = timeit.default_timer() - started
            return metrics_out

        muon_matrix_mask = unflatten_dict({
            name: w.ndim == 2 and name not in (
                'params.transformer.wte.embedding', 'params.lm_head.kernel')
            for name, w in flatten_dict(train_state.params, sep='.').items()
        }, sep='.')
        outer_decay_mask = jax.tree.map(
            lambda w: w.ndim == 2, train_state.params)

        def apply_outer_decay(base, candidate):
            if FLAGS.outer_weight_decay == 0.0:
                return candidate
            return jax.tree.map(
                lambda w, c, use: c - FLAGS.outer_weight_decay * w if use else c,
                base, candidate, outer_decay_mask)

        def muon_matrix_norm(params):
            return global_norm([
                w.astype(jnp.float32) for w, use in zip(
                    jax.tree.leaves(params), jax.tree.leaves(muon_matrix_mask)) if use])

        @jax.jit
        def outer_update_metrics(before, after, direction, alpha):
            wnorm = muon_matrix_norm(before)
            denom = jnp.maximum(wnorm, 1e-12)
            delta = jax.tree.map(lambda new, old: new - old, after, before)
            return {
                'outer/muon_weight_norm_before': wnorm,
                'outer/muon_weight_norm_after': muon_matrix_norm(after),
                'outer/muon_solver_relative_update': alpha * muon_matrix_norm(direction) / denom,
                'outer/muon_total_relative_update': muon_matrix_norm(delta) / denom,
                'outer/muon_decay_relative_update': FLAGS.outer_weight_decay * wnorm / denom,
                'outer/embedding_weight_norm_after': jnp.linalg.norm(
                    after['params']['transformer']['wte']['embedding'].astype(jnp.float32)),
                'outer/head_weight_norm_after': jnp.linalg.norm(
                    after['params']['lm_head']['kernel'].astype(jnp.float32)),
            }

        if FLAGS.optimizer_type == "cg":
            # Persistent Adam first and second moments for the CG path.
            cg_first_moment = jax.tree_util.tree_map(
                jnp.zeros_like,
                train_state.params,
            )
            cg_second_moment = jax.tree_util.tree_map(
                jnp.zeros_like,
                train_state.params,
            )
            cg_adam_step = jnp.array(0, dtype=jnp.int32)
            cg_x0 = jax.tree_util.tree_map(
                jnp.zeros_like,
                train_state.params,
            )

        if FLAGS.optimizer_type == "cg" and FLAGS.outer_momentum_beta > 0.0:
            outer_prev_update = jax.tree_util.tree_map(
                jnp.zeros_like,
                train_state.params,
            )

        if cg_metadata is not None:
            cg_first_moment = restored_cg['cg_first_moment']
            cg_second_moment = restored_cg['cg_second_moment']
            cg_x0 = restored_cg['cg_x0']
            cg_adam_step = jax.device_put(restored_cg['cg_adam_step'])
            sharded_rng = jax.device_put(restored_cg['sharded_rng'])
            if FLAGS.outer_momentum_beta > 0.0:
                outer_prev_update = restored_cg['outer_prev_update']
            if FLAGS.weight_average:
                ema = restored_cg['ema']
            del restored_cg

        if warmstart_params is not None and not FLAGS.reset_start:
            print('Using warmstart params')
            inner_state = inner_state.replace(params=warmstart_params)

        jax.block_until_ready((train_state, inner_state, sharded_rng))
        if FLAGS.log_initial_eval and FLAGS.eval_steps > 0 and cg_metadata is None:
            timing.start()
            initial_eval_metrics = []
            initial_eval_rng = jax.tree.map(lambda x: x.copy(), sharded_rng)
            with DatasetFactory.initial_eval_iterator(eval_dataset) as initial_eval_iterator:
                for _ in range(FLAGS.eval_steps):
                    eval_batch, _ = next(initial_eval_iterator)
                    initial_eval_rng, eval_metrics = sharded_eval_step(
                        train_state.params, initial_eval_rng, eval_batch)
                    initial_eval_metrics.append(eval_metrics)
            initial_record = progress.record(
                -1, **jax.device_get(average_metrics(initial_eval_metrics)))
            jax.block_until_ready((initial_eval_rng, initial_record))
            timing.stop_eval()
            initial_record.update(timing.metrics())
            wandb.log(initial_record, step=initial_record['completed_updates'], commit=True)

        if FLAGS.optimizer_type == 'cg' and FLAGS.save_model_freq > 0 and cg_metadata is None:
            progress.dataset_total_tokens = dataset_object.get_state_dict()['metadata']['dataset_total_tokens']
            save_checkpoint(train_state, ema=ema if FLAGS.weight_average else None)
        cg_checkpoint_safe = True
        for step in step_counter:
            cg_checkpoint_safe = False
            timing.start()
            k = FLAGS.train_batch_growth_interval
            solve_batch_size = FLAGS.train_dataset_batch_size
            if k > 0:
                solve_batch_size = min(
                    FLAGS.train_batch_max,
                    FLAGS.train_dataset_batch_size
                    + FLAGS.train_batch_growth_increment * (step // k),
                )
            actual_solve_batch_size = None
            pending_record = {}
            inner_diagnostic_rows = []

            def defer_wandb(values, *args, **kwargs):
                values = jax.device_get(values)
                if 'inner_step' in values:
                    inner_diagnostic_rows.append(dict(values))
                else:
                    pending_record.update(values)

            print("step", step, "param norm", global_norm(train_state.params), flush=True)

            if FLAGS.reset_start:
                inner_state = inner_state.replace(
                    params=train_state.params,
                    opt_state=tayl_solver.init(train_state.params)
                )

                if FLAGS.optimizer_type == "cg":
                    # Equivalent reset for cg
                    cg_x0 = jax.tree_util.tree_map(
                        jnp.zeros_like,
                        train_state.params,
                    )


            # Fetch only the solve batch below; no unused advance of the data cursor.
        
            # ------------------------------------------------------------------
            # Shared helpers, factored out of the original inline linesearch code
            # so the adaptive and non-adaptive paths use identical math.
            # ------------------------------------------------------------------
            def run_linesearch(base_params, dir, ls_batches, ls_rngs, init_step=None):
                losses = []
                if FLAGS.armijo_linesearch:
                    step_size = init_step if init_step is not None else FLAGS.armijo_init_step
                    best_loss = float("inf")
                    best_step_size = step_size
                    patience = FLAGS.patience
                    bad = 0
                    while step_size > 1e-6:
                        updated_params = jax.tree_util.tree_map(
                            lambda x, y: x + step_size * y, base_params, dir
                        )
                        updated_params = apply_outer_decay(base_params, updated_params)
                        accumulated_loss = 0.0
                        for batch, subrng in zip(ls_batches, ls_rngs):
                            loss, _ = microbatched_loss_fn(updated_params, batch, subrng, FLAGS.cg_n_micro)
                            accumulated_loss += loss
                        average_loss = float(jax.device_get(accumulated_loss / len(ls_batches)))
                        print(f"step={step_size:.6f}  loss={average_loss:.6f}")
                        losses.append((step_size, average_loss))
                        if average_loss < best_loss:
                            best_loss = average_loss
                            best_step_size = step_size
                            bad = 0
                        else:
                            bad += 1
                        if bad >= patience:
                            break
                        step_size *= FLAGS.armijo_beta
                    step_size = best_step_size
                    print(f"Chosen step size: {step_size:.6f}\n")
                else:
                    ls_candidates = [1 / jnp.sqrt(2) ** i for i in range(FLAGS.ls_range)]
                    for step_size in ls_candidates:
                        updated_params = jax.tree_util.tree_map(
                            lambda x, y: x + step_size * y, base_params, dir
                        )
                        updated_params = apply_outer_decay(base_params, updated_params)
                        accumulated_loss = 0.0
                        for batch, subrng in zip(ls_batches, ls_rngs):
                            loss, _ = microbatched_loss_fn(updated_params, batch, subrng, FLAGS.cg_n_micro)
                            accumulated_loss += loss
                        average_loss = accumulated_loss / len(ls_batches)
                        losses.append((step_size, average_loss))
                    step_size, _ = min(losses, key=lambda x: x[1])
                    step_size = jax.device_get(step_size)
                return step_size, losses

            def pull_ls_batches_and_baseline(sharded_rng, base_params, dataset):
                exit_flag = False
                num_ls_batches = FLAGS.ls_eval_batches if FLAGS.ls_eval_batches > 0 else FLAGS.inner_loop_iter
                ls_batches = []
                for _ in range(num_ls_batches):
                    try:
                        batch, metadata = pull_training_batch('linesearch')
                        ls_batches.append(batch)
                    except StopIteration:
                        print("Dataset exhausted")
                        exit_flag = True
                        break
                if exit_flag:
                    return None, None, sharded_rng, None, True
                ls_rngs = []
                for _ in ls_batches:
                    sharded_rng, subrng = jax.random.split(sharded_rng)
                    ls_rngs.append(subrng)
                baseline_loss = 0.0
                for batch, subrng in zip(ls_batches, ls_rngs):
                    bl, _ = microbatched_loss_fn(base_params, batch, subrng, FLAGS.cg_n_micro)
                    baseline_loss += bl
                baseline_loss = float(jax.device_get(baseline_loss / len(ls_batches)))
                return ls_batches, ls_rngs, sharded_rng, baseline_loss, False

            if FLAGS.single_batch_inner:
                single_batch_, single_dataset_metrics_ = pull_training_batch('solve')

            ADAPTIVE_CHECKPOINTS_ALL = [1, 4, 16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 512, 640, 768, 1024, 1280, 1536,1792, 2048, 2560]
            if FLAGS.optimizer_type == 'cg':
                if FLAGS.single_batch_inner:
                    batch_, dataset_metrics_ = single_batch_, single_dataset_metrics_
                else:
                    batch_, dataset_metrics_ = pull_training_batch('solve')
                batch = jax.tree.map(
                    lambda x: jax.lax.with_sharding_constraint(x, PS(('dp', 'fsdp'))),
                    batch_
                )
                scheduled_lambda = FLAGS.cg_interpolation_lambda
                if FLAGS.cg_lambda_batch_denominator > 0:
                    scheduled_lambda = cg_resume.batch_lambda(
                        batch_['input_tokens'].shape[0], FLAGS.cg_lambda_batch_denominator)
                elif FLAGS.cg_lambda_final != -1.0:
                    fraction = min(max(step / FLAGS.cg_lambda_ramp_steps, 0.0), 1.0)
                    scheduled_lambda += fraction * (
                        FLAGS.cg_lambda_final - FLAGS.cg_interpolation_lambda)
                (
                    candidate_params,
                    cg_first_moment,
                    cg_second_moment,
                    cg_x0,                 # NEW
                    cg_adam_step,
                    sharded_rng,
                    cg_metrics,
                ) = sharded_train_step_cg(
                    train_state.params,
                    cg_first_moment,
                    cg_second_moment,
                    cg_x0,                 # NEW
                    cg_adam_step,
                    train_state.step,
                    sharded_rng,
                    batch,
                    FLAGS.inner_loop_wd,
                    jnp.asarray(scheduled_lambda, dtype=jnp.float32),
                )
                if FLAGS.condition_log and step % FLAGS.condition_every == 0:
                    beta2_correction = 1.0 - FLAGS.optimizer.adamw_optimizer.b2 ** int(
                        jax.device_get(cg_adam_step))
                    adam_diagonal = jax.tree.map(
                        lambda moment: jnp.sqrt(moment / beta2_correction) + 1e-8,
                        cg_second_moment)
                    condition_metrics = run_condition_diagnostics(
                        train_state.params, batch, cg_diagonal=adam_diagonal,
                        effective_lambda=float(jax.device_get(
                            cg_metrics['cg_lambda_effective'])),
                        safe_adam_lr=max(float(jax.device_get(
                            cg_metrics['adamw_learning_rate'])), 1e-12))
                    defer_wandb(condition_metrics, step=step)
                
                ls_batches, ls_rngs, sharded_rng, baseline_loss, exit_flag = pull_ls_batches_and_baseline(
                    sharded_rng, train_state.params, dataset
                )
                if exit_flag:
                    jax.block_until_ready((train_state, inner_state, sharded_rng,
                                           cg_first_moment, cg_second_moment,
                                           cg_x0, cg_adam_step, cg_metrics))
                    timing.stop_train_interval(completed_update=False)
                    break
                print(f"\nTrue model loss: {baseline_loss:.6f}")

                dir = jax.tree_util.tree_map(lambda x, y: x - y, candidate_params, train_state.params)
                if FLAGS.normalize_step:
                    dir_norm_val = global_norm(dir)
                    dir = jax.tree_util.tree_map(lambda x: x / (dir_norm_val + 1e-8), dir)

                if FLAGS.outer_momentum_beta > 0.0:
                    raw_dir_norm = global_norm(dir)
                    prev_update_norm = global_norm(outer_prev_update)
                    dir = jax.tree_util.tree_map(
                        lambda d, prev: d + FLAGS.outer_momentum_beta * prev,
                        dir,
                        outer_prev_update,
                    )

                step_size, losses = run_linesearch(train_state.params, dir, ls_batches, ls_rngs)
                effective_step_size = FLAGS.fixed_step_size if FLAGS.fixed_step_size > 0.0 else step_size
                print("Step size:", effective_step_size)

                if FLAGS.outer_momentum_beta > 0.0:
                    accepted_update = jax.tree_util.tree_map(
                        lambda d: effective_step_size * d,
                        dir,
                    )
                    updated_params = jax.tree_util.tree_map(
                        lambda p, u: p + u,
                        train_state.params,
                        accepted_update,
                    )
                    outer_prev_update = accepted_update
                else:
                    updated_params = jax.tree_util.tree_map(
                        lambda p, d: p + effective_step_size * d,
                        train_state.params,
                        dir,
                    )
                train_state = train_state.replace(step=train_state.step + 1, params=updated_params)

                metrics = dict(cg_metrics)
                metrics['param_norm'] = global_norm(updated_params)

                if step % FLAGS.log_freq == 0:
                    dir_norm = float(jax.device_get(global_norm(dir)))
                    defer_wandb({
                        "step_size": effective_step_size,
                        "global_step": step,
                        "scaled_step_norm": effective_step_size * dir_norm,
                        "dir_norm": dir_norm,
                        "loss": baseline_loss,
                        **({
                            "raw_dir_norm": float(jax.device_get(raw_dir_norm)),
                            "momentum_dir_norm": dir_norm,
                            "prev_update_norm": float(jax.device_get(prev_update_norm)),
                            "accepted_update_norm": float(jax.device_get(global_norm(accepted_update))),
                            "outer_momentum_beta": FLAGS.outer_momentum_beta,
                        } if FLAGS.outer_momentum_beta > 0.0 else {}),
                    }, step=step)

            elif FLAGS.adaptive_inner_loop and FLAGS.linesearch:
                # ---------------- Adaptive checkpointed inner-loop search ----------------
                checkpoint_cap = min(FLAGS.inner_loop_iter, 2560)
                checkpoints = [c for c in ADAPTIVE_CHECKPOINTS_ALL if c <= checkpoint_cap]
                if not checkpoints or checkpoints[-1] != checkpoint_cap:
                    checkpoints.append(checkpoint_cap)

                best_inner_state = None   # full snapshot (params + opt_state) at best checkpoint
                best_step_size = None
                best_checkpoint = None
                prev_best_loss = float('inf')
                exit_training = False

                i = 0
                checkpoint_metrics = {}
                ls_batches, ls_rngs, sharded_rng, baseline_loss, exit_flag = pull_ls_batches_and_baseline(
                    sharded_rng, train_state.params, dataset
                )
                if exit_flag:
                    exit_training = True
                    checkpoints = []
                for checkpoint in checkpoints:
                    while i < checkpoint:
                        if FLAGS.single_batch_inner:
                            batch_, dataset_metrics_ = single_batch_, single_dataset_metrics_
                        else:
                            batch_, dataset_metrics_ = pull_training_batch('solve')
                        batch = jax.tree.map(
                            lambda x: jax.lax.with_sharding_constraint(x, PS(('dp', 'fsdp'))),
                            batch_
                        )
                        if (i == 0 and FLAGS.condition_log
                                and step % FLAGS.condition_every == 0):
                            # Deterministic diagnostic of the first already-fetched
                            # Muon solve batch; training RNG and cursor are untouched.
                            condition_metrics = run_condition_diagnostics(
                                train_state.params, batch)
                            defer_wandb(condition_metrics, step=step)
                        # is_last_step deliberately always False here -- see explanation
                        inner_state, sharded_rng, metrics = sharded_train_step(
                            inner_state, train_state.params, sharded_rng, batch,
                            FLAGS.inner_loop_wd, jnp.bool_((i + 1) == checkpoint)
                        )
                        i += 1
                        if i == 1 or i % 100 == 0 or i == checkpoint:
                            print(f"  inner step {i}/{checkpoint} (adaptive) done", flush=True)

                    checkpoint_metrics[checkpoint] = metrics
                        # if FLAGS.log_inner_steps:
                        #     log_metrics = {"inner_step": step*FLAGS.inner_loop_iter + i}
                        #     log_metrics['inner_loss'] = metrics['linear_model_loss']
                        #     log_metrics['inner_gradient_norm'] = metrics['gradient_norm']
                        #     log_metrics['inner_param_norm'] = metrics['param_norm']
                        #     log_metrics['inner_gpu_memory'] = metrics['gpu_memory']
                        #     log_metrics['inner_learning_rate'] = metrics['learning_rate']
                        #     defer_wandb(log_metrics)

                    dir = jax.tree_util.tree_map(lambda x, y: x - y, inner_state.params, train_state.params)
                    if FLAGS.normalize_step:
                        dir_norm_val = global_norm(dir)
                        dir = jax.tree_util.tree_map(lambda x: x / (dir_norm_val + 1e-8), dir)

                    init_step = float(2.0 / jnp.sqrt(float(checkpoint)))
                    step_size, losses = run_linesearch(train_state.params, dir, ls_batches, ls_rngs, init_step=init_step)
                    step_size = float(jax.device_get(step_size))
                    ckpt_best_loss = min(l for _, l in losses)
                    print(f"checkpoint={checkpoint} loss={ckpt_best_loss:.6f} step_size={step_size:.6f}", flush=True)

                    if ckpt_best_loss >= prev_best_loss:
                        break  # no improvement -- keep the previous checkpoint's snapshot
                    prev_best_loss = ckpt_best_loss
                    best_inner_state = jax.device_get(inner_state)       # full pytree: params + opt_state
                    best_step_size = step_size
                    best_checkpoint = checkpoint

                if exit_training:
                    jax.block_until_ready((train_state, inner_state, sharded_rng))
                    timing.stop_train_interval(completed_update=False)
                    break  # dataset exhausted; end training, same as the non-adaptive path

                dir = jax.tree_util.tree_map(lambda x, y: x - y, best_inner_state.params, train_state.params)
                updated_params = jax.tree_util.tree_map(lambda x, y: x + best_step_size * y, train_state.params, dir)
                train_state = train_state.replace(
                    step=train_state.step + 1,
                    opt_state=best_inner_state.opt_state,
                    params=updated_params,
                    warmstart_params=best_inner_state.params,
                )
                print(f"Chosen checkpoint: {best_checkpoint}, step_size: {best_step_size:.6f}", flush=True)
                metrics = checkpoint_metrics[best_checkpoint]  # so b_norm/relative_residual reflect the chosen checkpoint
                if step % FLAGS.log_freq == 0:
                    defer_wandb({
                        "chosen_inner_checkpoint": best_checkpoint,
                        "step_size": best_step_size,
                        "global_step": step,
                        "loss": baseline_loss,
                    }, step=step)
                if FLAGS.weight_average:
                    alpha = FLAGS.weight_average_decay
                    ema = jax.tree_util.tree_map(lambda x, y: alpha * x + (1 - alpha) * y, ema, updated_params)

            else:
                # ---------------- Existing (non-adaptive) behavior, unchanged math ----------------
                outer_params_before = train_state.params
                for i in range(FLAGS.inner_loop_iter):
                    if FLAGS.single_batch_inner:
                        batch_, dataset_metrics_ = single_batch_, single_dataset_metrics_
                    else:
                        batch_, dataset_metrics_ = pull_training_batch('solve')
                    batch = jax.tree.map(
                        lambda x: jax.lax.with_sharding_constraint(x, PS(('dp', 'fsdp'))),
                        batch_
                    )
                    if (i == 0 and FLAGS.condition_log
                            and step % FLAGS.condition_every == 0):
                        condition_metrics = run_condition_diagnostics(
                            train_state.params, batch)
                        defer_wandb(condition_metrics, step=step)
                    is_last_step = jnp.bool_((i + 1) == FLAGS.inner_loop_iter)
                    inner_state, sharded_rng, metrics = sharded_train_step(
                        inner_state, train_state.params, sharded_rng, batch, FLAGS.inner_loop_wd, is_last_step
                    )
                    if (i + 1) == 1 or (i + 1) % 100 == 0 or (i + 1) == FLAGS.inner_loop_iter:
                        print(f"  inner step {i+1}/{FLAGS.inner_loop_iter} done", flush=True)
                    if FLAGS.log_inner_steps:
                        log_metrics = {"inner_step": step*FLAGS.inner_loop_iter + i}
                        log_metrics['inner_loss'] = metrics['linear_model_loss']
                        log_metrics['inner_gradient_norm'] = metrics['gradient_norm']
                        log_metrics['inner_param_norm'] = metrics['param_norm']
                        log_metrics['inner_gpu_memory'] = metrics['gpu_memory']
                        log_metrics['inner_learning_rate'] = metrics['learning_rate']
                        defer_wandb(log_metrics)
                    if FLAGS.weight_average and not FLAGS.linesearch:
                        alpha = FLAGS.weight_average_decay
                        ema = jax.tree_util.tree_map(lambda x, y: alpha*x + (1-alpha)*y, ema, inner_state.params)

                if FLAGS.linesearch:
                    ls_batches, ls_rngs, sharded_rng, baseline_loss, exit_flag = pull_ls_batches_and_baseline(
                        sharded_rng, train_state.params, dataset
                    )
                    if exit_flag:
                        jax.block_until_ready(
                            (train_state, inner_state, sharded_rng, metrics))
                        timing.stop_train_interval(completed_update=False)
                        break
                    print(f"\nTrue model loss: {baseline_loss:.6f}")

                    dir = jax.tree_util.tree_map(lambda x, y: x - y, inner_state.params, train_state.params)
                    if FLAGS.normalize_step:
                        dir_norm_val = global_norm(dir)
                        dir = jax.tree_util.tree_map(lambda x: x / (dir_norm_val + 1e-8), dir)

                    step_size, losses = run_linesearch(train_state.params, dir, ls_batches, ls_rngs)

                    effective_step_size = FLAGS.fixed_step_size if FLAGS.fixed_step_size > 0.0 else step_size
                    print("Step size:", effective_step_size)
                    dir_norm = float(jax.device_get(global_norm(dir)))
                    defer_wandb({
                        "step_size": effective_step_size,
                        "global_step": step,
                        "scaled_step_norm": effective_step_size * dir_norm,
                        "dir_norm": dir_norm,
                        "loss": baseline_loss,
                        }, step=step)
                    for (_step_size, _loss) in losses:
                        tag = f"{_step_size:.4f}"
                        loss_improvement = baseline_loss - float(jax.device_get(_loss))
                        defer_wandb({
                            f"ls_loss_improvement_{tag}": loss_improvement,
                            "global_step": step,
                        }, step=step)

                    updated_params = jax.tree_util.tree_map(lambda x, y: x + effective_step_size*y, train_state.params, dir)
                    updated_params = apply_outer_decay(train_state.params, updated_params)
                    train_state = train_state.replace(
                        step=train_state.step+1,
                        opt_state=inner_state.opt_state,
                        params=updated_params,
                        warmstart_params=inner_state.params,
                    )
                    if FLAGS.weight_average:
                        alpha = FLAGS.weight_average_decay
                        ema = jax.tree_util.tree_map(lambda x, y: alpha*x + (1-alpha)*y, ema, updated_params)
                else:
                    dir = jax.tree.map(lambda inner, outer: inner - outer,
                                       inner_state.params, train_state.params)
                    effective_step_size = 1.0
                    updated_params = apply_outer_decay(train_state.params, inner_state.params)
                    train_state = train_state.replace(
                        step=train_state.step+1,
                        opt_state=inner_state.opt_state,
                        params=updated_params
                    )
                if FLAGS.optimizer_type == 'muon' and FLAGS.gauss_newton and step % FLAGS.log_freq == 0:
                    metrics.update(outer_update_metrics(
                        outer_params_before, train_state.params, dir, effective_step_size))
                del outer_params_before
       
            cg_checkpoint_safe = True
            progress.complete_update()
            live_results = [train_state, inner_state, sharded_rng, metrics]
            if FLAGS.optimizer_type == 'cg':
                live_results.extend((cg_first_moment, cg_second_moment,
                                     cg_x0, cg_adam_step))
            jax.block_until_ready(live_results)
            timing.stop_train_interval(completed_update=True)
            defer_wandb({'train_batch_size': actual_solve_batch_size}, step=step)
            if FLAGS.optimizer_type == 'cg':
                n_actual, groups = microbatch_groups(
                    actual_solve_batch_size, FLAGS.cg_n_micro, data_shards)
                local_sizes = [mb_size // data_shards
                               for _, count, mb_size in groups for _ in range(count)]
                defer_wandb({
                    'cg_n_micro_requested': FLAGS.cg_n_micro,
                    'cg_n_micro': n_actual,
                    'cg_microbatch_per_device_min': min(local_sizes),
                    'cg_microbatch_per_device_max': max(local_sizes),
                    'cg_lambda_scheduled': cg_metrics['cg_lambda_scheduled'],
                    'cg_lambda_effective': cg_metrics['cg_lambda_effective'],
                    'adamw_learning_rate': cg_metrics['adamw_learning_rate'],
                    'cg_relative_damping': cg_metrics['cg_relative_damping'],
                    **({'cg_relative_damping': None} if scheduled_lambda == 0.0 else {}),
                }, step=step)
            should_log = (
                step % FLAGS.log_freq == 0
                or FLAGS.train_batch_growth_interval > 0
                or FLAGS.optimizer_type == 'cg'
                or (FLAGS.condition_log
                    and step % FLAGS.condition_every == 0))
            if should_log:
                log_metrics = {}
                stop_after_log = False
                if step % FLAGS.log_freq == 0:
                    log_metrics.update(get_tpu_metrics())
                    log_metrics.update(metrics)
                    log_metrics["param_norm"] = global_norm(train_state.params)
                # log_metrics.update(dataset_metrics)

                do_eval = step % FLAGS.log_freq == 0 and FLAGS.eval_freq and FLAGS.eval_steps > 0 and ((step % FLAGS.eval_freq == 0 and step <= FLAGS.total_steps * 0.5) or (step % FLAGS.log_freq == 0 and step > FLAGS.total_steps * 0.5))

                if do_eval: # eval_freq must be | by log_freq
                    timing.start()
                    eval_iterator = iter(eval_dataset)
                    eval_metric_list = []
                    for _ in range(FLAGS.eval_steps):
                        eval_batch, _ = next(eval_iterator)

                        if FLAGS.weight_average:
                            eval_params=ema
                        else:
                            eval_params = train_state.params
                        sharded_rng, eval_metrics = sharded_eval_step(
                            eval_params, sharded_rng, eval_batch
                        )
                        eval_metric_list.append(eval_metrics)
                    if eval_metric_list:
                        log_metrics.update(average_metrics(eval_metric_list))
                    jax.block_until_ready((sharded_rng, log_metrics))
                    timing.stop_eval()
                    if FLAGS.target_loss > 0.0 and log_metrics['eval_loss'] <= FLAGS.target_loss:
                        print(f"Target loss {FLAGS.target_loss} reached with loss {log_metrics['eval_loss']}, stopping at step {step}")
                        log_metrics = jax.device_get(log_metrics)
                        defer_wandb(log_metrics)
                        tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")
                        
                        stop_after_log = True
                    elif FLAGS.target_loss > 0.0 and log_metrics['eval_loss'] >= 15:
                        print(f"Loss {log_metrics['eval_loss']} too high, stopping at step {step}")
                        stop_after_log = True
                    # metrics.update({"step": step})
                    # metrics = jax.device_get(metrics)
                    # logger.log(metrics)
                log_metrics = jax.device_get(log_metrics)
                log_metrics.update(pending_record)
                log_metrics.update(timing.metrics())
                log_metrics = progress.record(step, **log_metrics)
                if inner_diagnostic_rows:
                    log_metrics['inner_diagnostics'] = wandb.Table(
                        data=[[row] for row in inner_diagnostic_rows], columns=['metrics'])
                # Evaluation has consumed RNG; commit CG state before publishing the row.
                if FLAGS.optimizer_type == 'cg' and (
                        (FLAGS.save_model_freq > 0 and (step + 1) % FLAGS.save_model_freq == 0)
                        or (FLAGS.save_milestone_freq > 0 and (step + 1) % FLAGS.save_milestone_freq == 0)):
                    save_checkpoint(
                        train_state, ema=ema if FLAGS.weight_average else None,
                        milestone=(FLAGS.save_milestone_freq > 0
                                   and (step + 1) % FLAGS.save_milestone_freq == 0))
                wandb.log(log_metrics, step=log_metrics['completed_updates'], commit=True)
                tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")
                if stop_after_log:
                    break
            
            

            if FLAGS.optimizer_type != 'cg':
                if FLAGS.save_milestone_freq > 0 and (step + 1) % FLAGS.save_milestone_freq == 0:
                    if FLAGS.weight_average:
                        ema = jax.device_get(ema)
                        save_checkpoint(train_state, ema=ema, milestone=True)
                    else:
                        save_checkpoint(train_state, milestone=True)
                elif FLAGS.save_model_freq > 0 and (step + 1) % FLAGS.save_model_freq == 0:
                    if FLAGS.weight_average:
                        ema = jax.device_get(ema)
                        save_checkpoint(train_state, ema=ema)
                    else:
                        save_checkpoint(train_state)

        # Save before terminal evaluation consumes RNG, and never save a partial solve.
        if FLAGS.optimizer_type == 'cg' and FLAGS.save_model_freq > 0 and cg_checkpoint_safe:
            save_checkpoint(train_state, ema=ema if FLAGS.weight_average else None)
        terminal_metrics = {}
        if FLAGS.eval_freq != 0 and FLAGS.eval_steps > 0: # eval_freq must be | by log_freq
            timing.start()
            eval_iterator = iter(eval_dataset)
            eval_metric_list = []
            for _ in range(FLAGS.eval_steps):
                eval_batch, _ = next(eval_iterator)

                if FLAGS.weight_average:
                    eval_params=ema
                else:
                    eval_params = train_state.params
                sharded_rng, eval_metrics = sharded_eval_step(
                    eval_params, sharded_rng, eval_batch
                )
                eval_metric_list.append(eval_metrics)
            if eval_metric_list:
                terminal_metrics.update(average_metrics(eval_metric_list))
            terminal_metrics = jax.device_get(terminal_metrics)
            jax.block_until_ready((sharded_rng, terminal_metrics))
            timing.stop_eval()
        terminal_metrics.update(timing.metrics())
        terminal_record = progress.record(
            progress.phase_completed_updates - 1, **terminal_metrics)
        for name, value in terminal_record.items():
            wandb.run.summary[f'terminal_{name}'] = value
        if FLAGS.save_model_freq > 0 and FLAGS.optimizer_type != 'cg':
            save_checkpoint(train_state)

    wandb.finish()


if __name__ == "__main__":
    print(jax.local_devices())
    print(jax.devices())
    mlxu.run(main)
