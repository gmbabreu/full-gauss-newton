#!/usr/bin/env bash
set -euo pipefail

# Local helper: create/setup/run Figure-1 TPU experiments via gcloud.
#
# Example:
#   export PROJECT=fast-second-order
#   export ZONE=europe-west4-a
#   export TPU_NAME=fig1-v6e-8
#   export ACCELERATOR_TYPE=v6e-8
#   export REPO_URL=https://github.com/<you>/full-gauss-newton.git
#   export REPO_BRANCH=main
#   export WANDB_API_KEY=...
#   export OUTPUT_DIR=gs://<bucket>/figure1-runs
#   export TRAIN_PRETOKENIZED_DIR=gs://<bucket>/c4-tokenized
#   export EVAL_PRETOKENIZED_DIR=gs://<bucket>/c4-tokenized-val
#   bash scripts/gcloud_tpu_figure1.sh create
#   bash scripts/gcloud_tpu_figure1.sh setup
#   bash scripts/gcloud_tpu_figure1.sh configure
#   bash scripts/gcloud_tpu_figure1.sh run-both

ACTION="${1:-}"
PROJECT="${PROJECT:-fast-second-order}"
ZONE="${ZONE:-europe-west4-a}"
TPU_NAME="${TPU_NAME:-fig1-v6e-8}"
ACCELERATOR_TYPE="${ACCELERATOR_TYPE:-v6e-8}"
RUNTIME_VERSION="${RUNTIME_VERSION:-tpu-ubuntu2204-base}"
REPO_URL="${REPO_URL:-}"
REPO_BRANCH="${REPO_BRANCH:-main}"
REMOTE_DIR="${REMOTE_DIR:-full-gauss-newton}"

ssh_cmd() {
  gcloud compute tpus tpu-vm ssh "${TPU_NAME}" --zone="${ZONE}" --project="${PROJECT}" --command "$1"
}

case "${ACTION}" in
  create)
    gcloud config set project "${PROJECT}"
    gcloud compute tpus tpu-vm create "${TPU_NAME}" \
      --zone="${ZONE}" \
      --accelerator-type="${ACCELERATOR_TYPE}" \
      --version="${RUNTIME_VERSION}" \
      --spot
    ;;
  setup)
    : "${REPO_URL:?Set REPO_URL to your git repository URL}"
    ssh_cmd "set -eo pipefail; \
      if [ ! -d ${REMOTE_DIR} ]; then \
        git clone --branch ${REPO_BRANCH} ${REPO_URL} ${REMOTE_DIR}; \
      else \
        cd ${REMOTE_DIR}; git fetch origin; git checkout ${REPO_BRANCH}; git pull; cd ..; \
      fi; \
      cd ${REMOTE_DIR}; \
      export PYTHONPATH=\"\$PWD\${PYTHONPATH:+:\$PYTHONPATH}\"; \
      bash scripts/setup_tpu_vm.sh"
    ;;
  configure)
    : "${WANDB_API_KEY:?Set WANDB_API_KEY}"
    : "${OUTPUT_DIR:?Set OUTPUT_DIR}"
    : "${TRAIN_PRETOKENIZED_DIR:?Set TRAIN_PRETOKENIZED_DIR}"
    : "${EVAL_PRETOKENIZED_DIR:?Set EVAL_PRETOKENIZED_DIR}"
    ssh_cmd "cat > ${REMOTE_DIR}/.env.fig1 <<'ENVVARS'
export WANDB_API_KEY=${WANDB_API_KEY}
export OUTPUT_DIR=${OUTPUT_DIR}
export TRAIN_PRETOKENIZED_DIR=${TRAIN_PRETOKENIZED_DIR}
export EVAL_PRETOKENIZED_DIR=${EVAL_PRETOKENIZED_DIR}
ENVVARS"
    ;;
  run-adam)
    ssh_cmd "set -eo pipefail; cd ${REMOTE_DIR}; [ -f .venv/bin/activate ] && source .venv/bin/activate || true; source .env.fig1; bash templates/adam-tpu-vm-figure1-4000.sh"
    ;;
  run-gn)
    ssh_cmd "set -eo pipefail; cd ${REMOTE_DIR}; [ -f .venv/bin/activate ] && source .venv/bin/activate || true; source .env.fig1; bash templates/gn-tpu-vm-figure1-4000.sh"
    ;;
  run-both)
    "$0" run-adam
    "$0" run-gn
    ;;
  ssh)
    gcloud compute tpus tpu-vm ssh "${TPU_NAME}" --zone="${ZONE}" --project="${PROJECT}"
    ;;
  delete)
    gcloud compute tpus tpu-vm delete "${TPU_NAME}" --zone="${ZONE}" --project="${PROJECT}" --quiet
    ;;
  *)
    echo "Usage: $0 {create|setup|configure|run-adam|run-gn|run-both|ssh|delete}" >&2
    exit 1
    ;;
esac
