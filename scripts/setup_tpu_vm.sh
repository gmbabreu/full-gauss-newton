#!/usr/bin/env bash
set -euo pipefail

# Run this on a TPU VM host to install Python deps for EasyLM TPU training.
# Handles hosts where python3-venv/ensurepip are not preinstalled.

ensure_venv() {
  if python3 -m venv .venv >/dev/null 2>&1; then
    return 0
  fi

  echo "python3 -m venv failed; attempting to install python3-venv..."
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y python3-venv || sudo apt-get install -y python3.10-venv || true
  fi

  python3 -m venv .venv >/dev/null 2>&1
}

USE_VENV=1
if ! ensure_venv; then
  USE_VENV=0
  echo "WARNING: could not create virtualenv; falling back to user-site installs via pip --user"
fi

if [[ "${USE_VENV}" == "1" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PYTHON_BIN="python"
  PIP_CMD="python -m pip"
else
  PYTHON_BIN="python3"
  PIP_CMD="python3 -m pip"
fi

${PYTHON_BIN} -m ensurepip --upgrade >/dev/null 2>&1 || true
${PIP_CMD} install --upgrade pip wheel setuptools

if [[ "${USE_VENV}" == "1" ]]; then
  ${PIP_CMD} install "jax[tpu]==0.4.34" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
  ${PIP_CMD} install \
    flax==0.8.3 \
    optax==0.2.2 \
    transformers==4.41.0 \
    sentencepiece \
    datasets \
    "mlxu>=0.1.13" \
    einops \
    gcsfs \
    google-cloud-storage \
    wandb==0.17.4 \
    tqdm \
    matplotlib \
    seaborn
else
  ${PIP_CMD} install --user "jax[tpu]==0.4.34" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
  ${PIP_CMD} install --user \
    flax==0.8.3 \
    optax==0.2.2 \
    transformers==4.41.0 \
    sentencepiece \
    datasets \
    "mlxu>=0.1.13" \
    einops \
    gcsfs \
    google-cloud-storage \
    wandb==0.17.4 \
    tqdm \
    matplotlib \
    seaborn
fi

# Guardrail: keep JAX pinned in case any dependency attempts to upgrade it.
# 0.4.34 avoids PJRT option mismatch errors seen with older pins on newer TPU runtimes.
if [[ "${USE_VENV}" == "1" ]]; then
  ${PIP_CMD} install --upgrade --no-deps "jax[tpu]==0.4.34" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
else
  ${PIP_CMD} install --user --upgrade --no-deps "jax[tpu]==0.4.34" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
fi

if [[ "${USE_VENV}" == "1" ]]; then
  echo "TPU VM Python environment ready. Activate with: source .venv/bin/activate"
else
  echo "TPU VM Python environment ready via user-site packages (no .venv available)."
fi
