#!/usr/bin/env bash
set -euo pipefail

# Run this on a TPU VM host to install Python deps for EasyLM TPU training.

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools

# TPU JAX wheels
python -m pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Training/runtime deps used by this repository.
python -m pip install \
  flax==0.8.3 \
  git+https://github.com/deepmind/optax.git \
  transformers==4.41.0 \
  sentencepiece \
  datasets \
  "mlxu>=0.1.13" \
  einops \
  gcsfs \
  google-cloud-storage \
  neural-tangents \
  wandb==0.17.4 \
  tqdm \
  matplotlib \
  seaborn

echo "TPU VM Python environment ready. Activate with: source .venv/bin/activate"
