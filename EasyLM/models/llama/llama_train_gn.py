import pprint
from functools import partial

from google.cloud import storage

from tqdm import tqdm, trange
import numpy as np
import mlxu
import subprocess as sp

import timeit
import os
import wandb
from libtpu.sdk import monitoring as tpu_monitoring
import copy

import jax
import jax.numpy as jnp
from jax import linearize, linear_transpose
from jax.experimental.pjit import pjit
from jax.sharding import PartitionSpec as PS
from flax.training.train_state import TrainState
from transformers import AutoTokenizer
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.scipy.sparse.linalg import cg

import optax

from EasyLM.data import DatasetFactory
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
    load_dataset_state='',
    log_freq=50,
    log_inner_steps=False,
    save_model_freq=0,
    save_milestone_freq=0,
    eval_freq=0,
    eval_steps=0,
    gradient_accumulation_steps=1,   # Dead flag
    inner_loop_iter=100,
    tokenizer='openlm-research/open_llama_3b_v2',
    train_dataset_batch_size=8,
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
    cg_n_micro=1,   # microbatches for CG G; 1 = no microbatching (default, backward-compatible)
    cg_log_matrix_norms=False,
    cg_matrix_norm_frobenius_probes=4,
    cg_matrix_norm_power_iters=8,
)

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

    if not 0.0 <= FLAGS.outer_weight_decay < 1.0:
        raise ValueError("outer_weight_decay must satisfy 0 <= rho < 1")
    if FLAGS.outer_weight_decay and (
        FLAGS.optimizer_type != 'muon' or not FLAGS.gauss_newton
        or FLAGS.adaptive_inner_loop or FLAGS.weight_average
    ):
        raise ValueError("outer decay requires non-adaptive Muon-GN and weight_average=False")

    output_dir = os.path.join(FLAGS.output_dir, FLAGS.experiment_id)
    variant = mlxu.get_user_flags(FLAGS, FLAGS_DEF)
    flags_config_dict = mlxu.user_flags_to_config_dict(FLAGS, FLAGS_DEF)

    log_config = mlxu.flatten_config_dict(flags_config_dict)

    set_random_seed(FLAGS.seed)

    print(FLAGS.train_dataset)
    init_checkpoint_path = FLAGS.load_checkpoint

    if FLAGS.load_checkpoint.split('::')[-1].startswith('gs://'):
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
    if FLAGS.load_dataset_state != "" and mlxu.load_pickle(FLAGS.load_dataset_state) is not None:
        FLAGS.load_dataset_state = load_from_gcs(FLAGS.load_dataset_state, os.path.join(FLAGS.tmp_dir, 'dataset_state.pkl')) 

    tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer)
    dataset = DatasetFactory.load_dataset(FLAGS.train_dataset, tokenizer)
    if FLAGS.load_dataset_state != "" and mlxu.load_pickle(FLAGS.load_dataset_state) is not None:
        dataset.load_state_dict(mlxu.load_pickle(FLAGS.load_dataset_state))
        print('loaded dataset state', flush=True)

    if FLAGS.eval_steps > 0:
        eval_dataset = DatasetFactory.load_dataset(
            FLAGS.eval_dataset, dataset.tokenizer
        )
        eval_iterator = iter(eval_dataset)

    seq_length = dataset.seq_length
    llama_config = LLaMAConfigurator.finalize_config(FLAGS.llama)

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
    ):
        rng_generator = JaxRNG(rng)
        batch_ = with_sharding_constraint(batch, PS(('dp', 'fsdp')))

        # ── compute b_param = ∇_theta L = J^T (∇_f L) (the parameter-space gradient) ──
        # Split the batch into equal microbatches. Each microbatch loss is
        # mean-normalized internally, so averaging their J^T ∇_f L contributions
        # recovers the full-batch parameter-space gradient.
        n_micro = FLAGS.cg_n_micro

        batch_size = batch_['input_tokens'].shape[0]
        assert batch_size % n_micro == 0
        mb_size = batch_size // n_micro

        def b_param_body(i, carry):
            # Slice the i-th microbatch out of the full batch.
            start = i * mb_size
            input_mb = jax.lax.dynamic_slice_in_dim(
                batch_['input_tokens'], start, mb_size, axis=0
            )
            target_mb = jax.lax.dynamic_slice_in_dim(
                batch_['target_tokens'], start, mb_size, axis=0
            )
            mask_mb = jax.lax.dynamic_slice_in_dim(
                batch_['loss_masks'], start, mb_size, axis=0
            )

            def f_mb(p):
                # Run the model on the microbatch to get logits.
                # deterministic=True is safe because dropout/FCM are disabled in this config,
                # and avoids mutating JaxRNG inside the traced fori_loop.
                out = model.apply(
                    p,
                    input_mb,
                    deterministic=True,
                )
                return out.logits

            def scalar_loss_mb(logits):
                loss, _ = cross_entropy_loss_and_accuracy_with_weight_decay(
                    logits,
                    target_mb,
                    params0,
                    params0,
                    mask_mb,
                    weight_decay=wd,
                )
                return loss

            # Linearize the model at params0: this gives the current logits and a JVP
            # function for applying the parameter-space Jacobian.
            logits0_mb, jvp_fn_mb = linearize(f_mb, params0)

            # Compute the logit-space loss gradient (∇_f L), then apply J^T to obtain the
            # parameter-space gradient contribution for this microbatch.
            grad_Ly_mb = jax.grad(scalar_loss_mb)
            g0_mb = grad_Ly_mb(logits0_mb)
            jt_fn_mb = linear_transpose(jvp_fn_mb, params0)
            (b_mb,) = jt_fn_mb(g0_mb)

            # Accumulate into the running sum. Division by n_micro happens after
            # the loop to keep the carry parameter-sized (not scaled-parameter-sized).
            return jax.tree_util.tree_map(
                lambda accumulated, contribution: accumulated + contribution,
                carry,
                b_mb,
            )
        # Apply jax loop
        b_param_sum = jax.lax.fori_loop(
            0,
            n_micro,
            b_param_body,
            jax.tree_util.tree_map(jnp.zeros_like, params0),
        )
        # Average: each microbatch loss is mean-normalized over mb_size, so
        # averaging n_micro microbatch gradients recovers the full-batch gradient.
        b_param = jax.tree_util.tree_map(
            lambda b: b / n_micro,
            b_param_sum,
        )

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

        interpolation_lambda = jnp.asarray(
            FLAGS.cg_interpolation_lambda,
            dtype=jnp.float32,
        )

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
            """
            Apply the raw GN operator G(v), accumulating microbatches with a
            parameter-sized lax.fori_loop carry.
            """

            def gn_body(i, carry):
                # Slice microbatch i from the full batch.
                start = i * mb_size
                input_mb = jax.lax.dynamic_slice_in_dim(
                    batch_['input_tokens'], start, mb_size, axis=0
                )
                target_mb = jax.lax.dynamic_slice_in_dim(
                    batch_['target_tokens'], start, mb_size, axis=0
                )
                mask_mb = jax.lax.dynamic_slice_in_dim(
                    batch_['loss_masks'], start, mb_size, axis=0
                )

                def f_mb(p):
                    # Run the model on the microbatch to get logits.
                    # deterministic=True is safe because dropout/FCM are disabled in this config,
                    # and avoids mutating JaxRNG inside the traced fori_loop
                    out = model.apply(
                        p,
                        input_mb,
                        deterministic=True,
                    )
                    return out.logits

                def scalar_loss_mb(logits):
                    loss, _ = cross_entropy_loss_and_accuracy_with_weight_decay(
                        logits,
                        target_mb,
                        params0,
                        params0,
                        mask_mb,
                        weight_decay=wd,
                    )
                    return loss

                # Linearize the model at params0 to obtain J_mb and the current logits.
                logits0_mb, jvp_fn_mb = linearize(f_mb, params0)
                grad_Ly_mb = jax.grad(scalar_loss_mb)
                jt_fn_mb = linear_transpose(jvp_fn_mb, params0)

                # Compute the GN-vector product J^T H J v:
                #   J v        -> forward-mode JVP
                #   H(J v)     -> Hessian-vector product in logit space
                #   J^T H J v  -> transpose JVP
                logits_v_mb = jvp_fn_mb(v)
                _, Hv_mb = jax.jvp(
                    grad_Ly_mb,
                    (logits0_mb,),
                    (logits_v_mb,),
                )
                (Gv_mb,) = jt_fn_mb(Hv_mb)

                return jax.tree_util.tree_map(
                    lambda accumulated, contribution: accumulated + contribution,
                    carry,
                    Gv_mb,
                )
            # Apply microbatch loop
            gn_sum = jax.lax.fori_loop(
                0,
                n_micro,
                gn_body,
                jax.tree_util.tree_map(jnp.zeros_like, params0),
            )
            # Average the mean-normalized microbatch GN contributions to recover
            # the full-batch GN-vector product.
            Gv_param = jax.tree_util.tree_map(
                lambda x: x / n_micro,
                gn_sum,
            )
            return Gv_param
        
        matrix_norm_metrics = {}
        if FLAGS.cg_log_matrix_norms:
            assert FLAGS.cg_matrix_norm_frobenius_probes >= 1
            assert FLAGS.cg_matrix_norm_power_iters >= 1

            param_leaves, param_treedef = jax.tree_util.tree_flatten(params0)
            num_param_leaves = len(param_leaves)

            def diagnostic_tree_dot(left, right):
                leaf_products = [
                    jnp.sum(x.astype(jnp.float32) * y.astype(jnp.float32))
                    for x, y in zip(
                        jax.tree_util.tree_leaves(left),
                        jax.tree_util.tree_leaves(right),
                    )
                ]
                return jnp.sum(jnp.stack(leaf_products))

            def diagnostic_tree_norm(tree):
                return jnp.sqrt(jnp.maximum(diagnostic_tree_dot(tree, tree), 0.0))

            def rademacher_tree(key):
                keys = jax.random.split(key, num_param_leaves)
                leaves = [
                    jax.random.rademacher(key, leaf.shape, dtype=leaf.dtype)
                    for key, leaf in zip(keys, param_leaves)
                ]
                return jax.tree_util.tree_unflatten(param_treedef, leaves)

            diagnostic_rng = jax.random.PRNGKey(FLAGS.seed)
            frobenius_rng, power_rng = jax.random.split(diagnostic_rng)

            def frobenius_body(i, squared_norm_sum):
                probe = rademacher_tree(jax.random.fold_in(frobenius_rng, i))
                g_probe = apply_G(probe)
                return squared_norm_sum + diagnostic_tree_dot(g_probe, g_probe)

            g_frob_squared = jax.lax.fori_loop(
                0,
                FLAGS.cg_matrix_norm_frobenius_probes,
                frobenius_body,
                jnp.asarray(0.0, dtype=jnp.float32),
            ) / jnp.asarray(
                FLAGS.cg_matrix_norm_frobenius_probes, dtype=jnp.float32
            )
            g_frob = jnp.sqrt(jnp.maximum(g_frob_squared, 0.0))

            power_vector = rademacher_tree(power_rng)
            power_vector_norm = diagnostic_tree_norm(power_vector)
            power_vector = jax.tree_util.tree_map(
                lambda value: value / (power_vector_norm + 1e-12),
                power_vector,
            )

            def power_body(_, vector):
                g_vector = apply_G(vector)
                g_vector_norm = diagnostic_tree_norm(g_vector)
                return jax.tree_util.tree_map(
                    lambda value: value / (g_vector_norm + 1e-12),
                    g_vector,
                )

            power_vector = jax.lax.fori_loop(
                0,
                FLAGS.cg_matrix_norm_power_iters - 1,
                power_body,
                power_vector,
            )
            g_power_vector = apply_G(power_vector)
            g_spectral = (
                diagnostic_tree_dot(power_vector, g_power_vector)
                / diagnostic_tree_dot(power_vector, power_vector)
            )
            power_residual = jax.tree_util.tree_map(
                lambda gq, q: gq - g_spectral * q,
                g_power_vector,
                power_vector,
            )
            g_spectral_relative_residual = (
                diagnostic_tree_norm(power_residual)
                / (diagnostic_tree_norm(g_power_vector) + 1e-12)
            )

            adam_diag = jax.tree_util.tree_map(
                lambda second_moment: (
                    jnp.sqrt(second_moment.astype(jnp.float32) / beta2_correction)
                    + adam_eps
                ),
                new_second_moment,
            )
            d_diag = jax.tree_util.tree_map(
                lambda diagonal: diagonal / safe_adam_lr,
                adam_diag,
            )
            d_leaves = jax.tree_util.tree_leaves(d_diag)
            d_frob = jnp.sqrt(jnp.sum(jnp.stack([
                jnp.sum(diagonal * diagonal) for diagonal in d_leaves
            ])))
            d_max_eig = jnp.max(jnp.stack([
                jnp.max(diagonal) for diagonal in d_leaves
            ]))
            d_min_eig = jnp.min(jnp.stack([
                jnp.min(diagonal) for diagonal in d_leaves
            ]))
            d_spectral = d_max_eig
            d_condition = jnp.where(
                d_min_eig > 0.0,
                d_max_eig / d_min_eig,
                jnp.asarray(jnp.inf, dtype=jnp.float32),
            )
            g_d_ratio_frob = g_frob / (d_frob + 1e-12)
            g_d_ratio_spec = g_spectral / (d_spectral + 1e-12)
            lambda_balance_frob = d_frob / (g_frob + d_frob + 1e-12)
            lambda_balance_spec = (
                d_spectral / (g_spectral + d_spectral + 1e-12)
            )

            interpolation_lambda = jnp.asarray(
                lambda_balance_spec*FLAGS.cg_interpolation_lambda,
                dtype=jnp.float32,
            )
            matrix_norm_metrics = {
                'G_frob': g_frob,
                'G_spectral': g_spectral,
                'G_spectral_relative_residual': g_spectral_relative_residual,
                'D_frob': d_frob,
                'D_spectral': d_spectral,
                'D_min_eig': d_min_eig,
                'D_max_eig': d_max_eig,
                'D_condition': d_condition,
                'G_D_ratio_frob': g_d_ratio_frob,
                'G_D_ratio_spec': g_d_ratio_spec,
                'cg_lambda_balance_frob': lambda_balance_frob,
                'cg_lambda_balance_spec': interpolation_lambda,
            }



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
            **matrix_norm_metrics,
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
        if n_micro == 1:
            loss, acc = parallel_loss_fn(params, batch, rng)
            return float(jax.device_get(loss)), float(jax.device_get(acc))
        batch_size = batch['input_tokens'].shape[0]
        assert batch_size % n_micro == 0, (
            f"Linesearch batch size {batch_size} must be divisible by cg_n_micro={n_micro}"
        )
        mb_size = batch_size // n_micro
        total_loss = 0.0
        total_acc  = 0.0
        rng_key = rng
        for i in range(n_micro):
            rng_key, subrng = jax.random.split(rng_key)
            mb = {
                'input_tokens':  batch['input_tokens'][i*mb_size:(i+1)*mb_size],
                'target_tokens': batch['target_tokens'][i*mb_size:(i+1)*mb_size],
                'loss_masks':    batch['loss_masks'][i*mb_size:(i+1)*mb_size],
            }
            loss, acc = parallel_loss_fn(params, mb, subrng)
            total_loss += float(jax.device_get(loss))
            total_acc  += float(jax.device_get(acc))
        return total_loss / n_micro, total_acc / n_micro

    def save_checkpoint(train_state, ema=None, milestone=False):
        step = int(jax.device_get(train_state.step))
        metadata = dict(
            step=step,
            variant=variant,
            flags=flags_config_dict,
            llama_config=llama_config.to_dict(),
        )
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
    print(f"Mesh axes names: {mesh.axis_names}")
    print(f"Mesh shape: {mesh.shape}")

    with mesh:
        print(mesh)
        train_state, restored_params = None, None
        warmstart_params = None
        if FLAGS.load_checkpoint != '':
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

        if FLAGS.wandb_run_id:
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


        start_step = int(jax.device_get(train_state.step))
        
        def copy_array(x):
            return copy.copy(x)  # or x.copy() if x is a NumPy/JAX array

        if FLAGS.save_model_freq > 0:
            if FLAGS.weight_average:
                ema = jax.tree.map(copy_array, train_state.params)
                save_checkpoint(train_state, ema=ema)
            else:
                save_checkpoint(train_state)

        sharded_rng = next_rng()

        step_counter = trange(start_step, FLAGS.total_steps, ncols=0)

        assert FLAGS.train_dataset_batch_size % mesh.shape['dp'] == 0, \
            "Batch size must be divisible by the number of devices in 'dp'."
        
        
        
        if FLAGS.weight_average:
            print('Using weight average')
            ema = jax.tree.map(copy_array, train_state.params)


        inner_state = create_trainstate_from_params(train_state.params)
        dataset = iter(dataset)

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

        if warmstart_params is not None and not FLAGS.reset_start:
            print('Using warmstart params')
            inner_state = inner_state.replace(params=warmstart_params)

        for step in step_counter:
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


            if FLAGS.single_batch_inner:
                single_batch_, single_dataset_metrics_ = next(dataset)
        
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
                        batch, _ = next(dataset)
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
                single_batch_, single_dataset_metrics_ = next(dataset)

            ADAPTIVE_CHECKPOINTS_ALL = [1, 4, 16, 32, 48, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 512, 640, 768, 1024, 1280, 1536,1792, 2048, 2560]
            if FLAGS.optimizer_type == 'cg':
                if FLAGS.single_batch_inner:
                    batch_, dataset_metrics_ = single_batch_, single_dataset_metrics_
                else:
                    batch_, dataset_metrics_ = next(dataset)
                batch = jax.tree.map(
                    lambda x: jax.lax.with_sharding_constraint(x, PS(('dp', 'fsdp'))),
                    batch_
                )
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
                )
                
                ls_batches, ls_rngs, sharded_rng, baseline_loss, exit_flag = pull_ls_batches_and_baseline(
                    sharded_rng, train_state.params, dataset
                )
                if exit_flag:
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
                    wandb.log({
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
                            batch_, dataset_metrics_ = next(dataset)
                        batch = jax.tree.map(
                            lambda x: jax.lax.with_sharding_constraint(x, PS(('dp', 'fsdp'))),
                            batch_
                        )
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
                        #     wandb.log(log_metrics)

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
                    wandb.log({
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
                        batch_, dataset_metrics_ = next(dataset)
                    batch = jax.tree.map(
                        lambda x: jax.lax.with_sharding_constraint(x, PS(('dp', 'fsdp'))),
                        batch_
                    )
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
                        wandb.log(log_metrics)
                    if FLAGS.weight_average and not FLAGS.linesearch:
                        alpha = FLAGS.weight_average_decay
                        ema = jax.tree_util.tree_map(lambda x, y: alpha*x + (1-alpha)*y, ema, inner_state.params)

                if FLAGS.linesearch:
                    ls_batches, ls_rngs, sharded_rng, baseline_loss, exit_flag = pull_ls_batches_and_baseline(
                        sharded_rng, train_state.params, dataset
                    )
                    if exit_flag:
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
                    wandb.log({
                        "step_size": effective_step_size,
                        "global_step": step,
                        "scaled_step_norm": effective_step_size * dir_norm,
                        "dir_norm": dir_norm,
                        "loss": baseline_loss,
                        }, step=step)
                    for (_step_size, _loss) in losses:
                        tag = f"{_step_size:.4f}"
                        loss_improvement = baseline_loss - float(jax.device_get(_loss))
                        wandb.log({
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
       
            if step % FLAGS.log_freq == 0:
                log_metrics = {"global_step": step}
                log_metrics.update(get_tpu_metrics())
                log_metrics.update(metrics)
                log_metrics["param_norm"] = global_norm(train_state.params)
                # log_metrics.update(dataset_metrics)
                # Token consumption logging (distinct tokens only)
                batches_per_step = 1 if FLAGS.single_batch_inner else FLAGS.inner_loop_iter
                tokens_per_step = FLAGS.train_dataset_batch_size * seq_length * batches_per_step
                log_metrics["total_tokens"] = tokens_per_step * (step + 1)
                

                do_eval = FLAGS.eval_freq and FLAGS.eval_steps > 0 and ((step % FLAGS.eval_freq == 0 and step <= FLAGS.total_steps * 0.5) or (step % FLAGS.log_freq == 0 and step > FLAGS.total_steps * 0.5))

                if do_eval: # eval_freq must be | by log_freq
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
                    if FLAGS.target_loss > 0.0 and log_metrics['eval_loss'] <= FLAGS.target_loss:
                        print(f"Target loss {FLAGS.target_loss} reached with loss {log_metrics['eval_loss']}, stopping at step {step}")
                        log_metrics = jax.device_get(log_metrics)
                        wandb.log(log_metrics)
                        tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")
                        
                        break
                    elif FLAGS.target_loss > 0.0 and log_metrics['eval_loss'] >= 15:
                        print(f"Loss {log_metrics['eval_loss']} too high, stopping at step {step}")
                        break
                    # metrics.update({"step": step})
                    # metrics = jax.device_get(metrics)
                    # logger.log(metrics)
                log_metrics = jax.device_get(log_metrics)
                wandb.log(log_metrics)
                tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")
            
            

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

        if FLAGS.eval_freq != 0 and FLAGS.eval_steps > 0: # eval_freq must be | by log_freq
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
            log_metrics = {"global_step": start_step}
            if eval_metric_list:
                log_metrics.update(average_metrics(eval_metric_list))
            log_metrics = jax.device_get(log_metrics)
            wandb.log(log_metrics)
            tqdm.write("\n" + pprint.pformat(log_metrics) + "\n")
        if FLAGS.save_model_freq > 0:
            save_checkpoint(train_state)

    wandb.finish()


if __name__ == "__main__":
    print(jax.local_devices())
    print(jax.devices())
    mlxu.run(main)
