# Checkpointing
To facilitate training very large language models that does not fit into the
main memory of a single machine, EasyLM adopt a streaming format of model
checkpoint. The streaming checkpointing format is implemented in
[checkpoint.py](/EasyLM/checkpoint.py). During checkpointing, the
StreamingCheckpointer simply flatten a nested state dictionary into a single
level dictionary, and stream the key, value pairs to a file one by one using
messagepack. Because it streams the tensors one by one, the checkpointer only
needs to gather one tensor from the distributed accelerators to the main memory
at a time, hence saving a lot of memory.


## Loading Checkpoint
While EasyLM mainly uses the streaming checkpointing format, it also supports
directly loading the standard flax checkpoint file created using
`flax.training.checkpoints.save_checkpoint`. The loading format can be specified
as part of the path passed into the training or serving script. For example, if
we want to serve a LLaMA model using the streaming checkpointing format, we can
use the following command:

``` shell
python -m EasyLM.models.llama.llama_serve \
    --load_checkpoint='params::path/to/checkpoint'
    ...
```

Note that the `params::` prefix is used to specify that the checkpoint is in
streaming format. The following prefix are supported for loading checkpoint:
* `params::`: Streaming checkpointing format.
* `flax::`: Standard flax checkpointing format.
* `trainstate::`: Loading an entire train state with optimizer state, this
    option is only supported for training script.
* `trainstate_params::`: Loading the params part from the entire train state.

By default, EasyLM does not save the optimizer state in the checkpoint, so
we will rarely need to use the `trainstate::` or `trainstate_params::` options.


## Streaming replay with first-order Adam

`llama_train.py` can repeatedly pack the first K raw-HF training batches with
`--train_dataset.huggingface_dataset.repeat_first_n_batches=K` (default 0).
This requires fixed batches, `tokens_count_at_start=0`, first-order `adamw`,
gradient accumulation 1, and no weight averaging. Validation replay stays 0.
The seeded HF stream restarts after K batches, discarding packing remainders;
inputs, shifted targets and loss masks repeat without packing across epochs.
No batch cache is created. The source must supply K complete packed batches.

To fork an eligible single-pass Adam checkpoint at step1000, retain **all**
parent model, tokenizer, source, shuffle, batch-size and optimizer flags
(including the learning-rate schedule), and add/override:

```shell
python -m EasyLM.models.llama.llama_train \
    --load_checkpoint='trainstate::PARENT/streaming_train_state_1000' \
    --load_dataset_state='' \
    --train_dataset.huggingface_dataset.repeat_first_n_batches=1000 \
    --train_dataset.huggingface_dataset.tokens_count_at_start=0 \
    --eval_dataset.huggingface_dataset.repeat_first_n_batches=0 \
    --optimizer.accumulate_gradient_steps=1 \
    --total_steps=50000 \
    --target_loss=0.0 \
    --dtype=fp32 --param_dtype=fp32 \
    --weight_average=False \
    --checkpointer.save_optimizer_state=True \
    --checkpointer.float_dtype=fp32 \
    --output_dir=OUTPUT_ROOT --experiment_id=NEW_REPLAY_EXPERIMENT \
    --wandb_run_id=''
```

`PARENT` must contain matching model, metadata, dataset and completion
snapshots, with full FP32 Adam state and RNG. The dataset companion is selected
automatically. The initial transition only permits enabling replay, increasing
`total_steps`, and changing `target_loss` to 0. It requires a new experiment
directory and W&B run. A parent may have consumed at most K batches: a
single-pass step1500 parent is rejected for K=1000. Old snapshots without the
repeat field in either training or validation config are treated as repeat=0.
Parameters-only continuation is unsupported for first-order Adam replay.

The complete train state (step, moments and schedule counters) and RNG are
restored. At the step1000 boundary, the next update consumes original batch0.
Mid-epoch recovery uses the saved document cursor and packing buffers.
Recovery of an already-enabled replay checkpoint retains strict trajectory
config matching; it does not permit another change to the training horizon.

`dataset_total_tokens` remains cumulative token visits. Replay adds
`replay_completed_epochs`, `replay_next_batch_index` (zero-based), and
`unique_solve_tokens`, capped at K * batch_size * sequence_length. For K=1000,
batch size 720 and sequence length 1024, this cap is 737,280,000 input positions.
Step50000 represents 50 total passes, including the inherited first pass.

## Adam-GN solve-stream replay

`llama_train_gn.py` supports replay for non-adaptive Adam-GN with
`single_batch_inner=True`, raw HF streaming, fixed matching batch sizes, and
no batch growth. Use an explicit positive `ls_eval_batches` (1 for this
experiment); replay rejects the legacy 0/inner_loop_iter fallback. Replay is
disabled for validation. Other solver paths remain unsupported for replay.

Retain the Adam experiment's exact tokenizer, source, text-processor/packing
flags, and GN model/solver settings. Append/override these flags on its GN
launch command (use a new output experiment and W&B run):

```shell
python -m EasyLM.models.llama.llama_train_gn \
    --load_checkpoint='trainstate_params::gs://gfn-soo/checkpoints/warmup_150m/1_0/streaming_train_state' \
    --load_dataset_state='' --cg_resume_state='' \
    --log_step_offset=0 --log_token_offset=0 --start_tokens=0 \
    --optimizer_type=adamw --gauss_newton=True \
    --single_batch_inner=True --adaptive_inner_loop=False \
    --train_dataset.type=huggingface \
    --train_dataset.huggingface_dataset.pretokenized_dataset_dir='' \
    --train_dataset.huggingface_dataset.streaming=True \
    --train_dataset.huggingface_dataset.batch_size=720 \
    --train_dataset_batch_size=720 --train_batch_growth_interval=0 \
    --train_dataset.huggingface_dataset.seq_length=1024 \
    --train_dataset.huggingface_dataset.shuffle_data=True \
    --train_dataset.huggingface_dataset.shuffle_seed=42 \
    --train_dataset.huggingface_dataset.shuffle_buffer_size=100000 \
    --train_dataset.huggingface_dataset.repeat_first_n_batches=500 \
    --train_dataset.huggingface_dataset.tokens_count_at_start=0 \
    --eval_dataset.huggingface_dataset.repeat_first_n_batches=0 \
    --linesearch=True --ls_eval_batches=1 \
    --total_steps=10000 --target_loss=0.0 \
    --save_model_freq=0 --save_milestone_freq=0
```

The checkpoint supplies parameters only; normal fresh GN solver initialization
and the existing reset policy, schedule and RNG progression apply. One solve
batch is fetched per outer update and reused throughout the inner solve. Only
the data iterator resets at epoch boundaries. There is no GN replay resume
system; dataset restoration, nonzero reporting offsets and saving are rejected
in this initial mode.

Line search uses a separate single-pass loader from a deep copy of the solve
config, changing only repeat to 0 and shuffle seed to solve_seed+1 (43).
Candidate step sizes reuse the fetched line-search batches and RNGs as before.
Changing ls_eval_batches does not advance the solve stream. Validation retains
its own loader. With replay disabled, solve and line search retain the original
shared training iterator and accounting behavior.

Outer-update records include solve-stream replay metadata. Line-search token
visits remain in phase_linesearch_tokens and cannot overwrite the solve dataset
cursor. After update501, the solve batch is original batch0, total_tokens is
369,377,280, unique_solve_tokens is 368,640,000, replay_completed_epochs is 1,
and replay_next_batch_index is 1. Step10000 completes 20 solve epochs.

## Saving Checkpoint
EasyLM will only save the checkpoint in the streaming format. By default, only
the model parameters are saved in the checkpoint file in the bfloat16 data type.
To configure the checkpointing behavior, you can use the following options:
* `float_dtype`: The float data type of the model parameters in the checkpoint file.
    The default value is `bf16`, other supported values are `fp32` and `fp16`.
* `save_optimizer_state`: Whether to save the entire train state in the checkpoint

Typically, we pass these optiosn into the training script. For example, for
LLaMA, we can use the following command to save the checkpoint in the fp32 data:
``` shell
python -m EasyLM.models.llama.llama_train \
    --checkpointer.float_dtype='fp32' \
    ...
```


## Converting Checkpoint to and from Standard Flax Format
To facilitate the use of EasyLM trained models with other Flax based libraries,
EasyLM provides a script to convert between the streaming checkpointing format
and the standard flax checkpointing format. The script can be found at
[EasyLM/scripts/convert_checkpoint.py](/EasyLM/scripts/convert_checkpoint.py).

To convert a checkpoint from the streaming format to the standard flax format,
use the following command:

``` shell
python -m EasyLM.scripts.convert_checkpoint \
    --load_checkpoint='params::path/to/checkpoint' \
    --output_file='path/to/output/checkpoint' \
    --streaming=False
```

To convert a standard flax checkpoint to the streaming format, use the following
command:

``` shell
python -m EasyLM.scripts.convert_checkpoint \
    --load_checkpoint='flax::path/to/checkpoint' \
    --output_file='path/to/output/checkpoint' \
    --streaming=True
```


## Diffing Checkpoint
To facilitate the release of fine-tuned model checkpoints that's based on
a non-public base model checkpoint, EasyLM provides a script to compute the
difference between two checkpoints. The script can be found at
[EasyLM/scripts/diff_checkpoint.py](/EasyLM/scripts/diff_checkpoint.py).

To compute the difference between a based checkpoint (based model) and a
target checkpoint (fine-tuned model), use the following command:

``` shell
python -m EasyLM.scripts.diff_checkpoint \
    --recover_diff=False \
    --load_base_checkpoint='params::path/to/based/checkpoint' \
    --load_target_checkpoint='params::path/to/target/checkpoint' \
    --output_file='path/to/output/checkpoint' \
    --streaming=True
```

The script will output a checkpoint that contains the difference between the
two checkpoints. You can use the `--streaming` flag to specify the format
(streaming or standard flax) of the output checkpoint. To recover a checkpoint
from a based checkpoint and a diff checkpoint, use the following command:

``` shell
python -m EasyLM.scripts.diff_checkpoint \
    --recover_diff=True \
    --load_base_checkpoint='params::path/to/base/checkpoint' \
    --load_target_checkpoint='params::path/to/diff/checkpoint' \
    --output_file='path/to/output/checkpoint' \
    --streaming=True
```
