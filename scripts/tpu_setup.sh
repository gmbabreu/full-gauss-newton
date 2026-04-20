#!/bin/bash
# TPU VM environment setup for full-gauss-newton experiments.
# Run once on a fresh TPU VM (tpu-ubuntu2204-base image).
# Pinned versions are known-good on JAX 0.4.30 / v4 & v5e TPUs.

set -e

echo "=== Installing JAX with TPU support ==="
pip install "jax[tpu]==0.4.30" \
    -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

echo "=== Installing pinned dependencies ==="
pip install \
    "flax==0.8.0" \
    "optax==0.2.2" \
    "orbax-checkpoint==0.6.4" \
    "transformers==4.40.0" \
    "neural-tangents==0.6.5" \
    "wandb==0.17.4" \
    "datasets" \
    "mlxu>=0.1.13" \
    "einops" \
    "gcsfs" \
    "google-cloud-storage" \
    "sentencepiece" \
    "tqdm" \
    "scipy" \
    "requests"

echo "=== Verifying JAX sees TPU devices ==="
python3 -c "import jax; print('Devices:', jax.devices())"

echo "=== Import check ==="
python3 -c "
import EasyLM.optimizers
import EasyLM.checkpoint
import EasyLM.data
import EasyLM.models.llama.llama_model
print('all imports ok')
"

echo "=== Setup complete ==="
