#!/bin/bash
# Provision three v4-8 TPU VMs and launch experiments in parallel.
# Each VM is single-host (4 chips) — no JAX distributed init needed.
# Training runs inside tmux so it survives SSH disconnection.
#
# Usage:
#   bash scripts/provision_tpu.sh

set -e

PROJECT="fast-second-order"
ZONE="europe-west4-a"
TPU_TYPE="v6e-8"
TPU_VERSION="v2-alpha-tpuv6e"
REPO="https://github.com/gmbabreu/full-gauss-newton.git"

WANDB_API_KEY="${WANDB_API_KEY:-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}"
WANDB_ENTITY="${WANDB_ENTITY:-ga2740-new-york-university}"
WANDB_PROJECT="${WANDB_PROJECT:-Gauss Newton}"
GCS_BUCKET="${GCS_BUCKET:-gs://fast-second-order-data}"

gcloud config set project "$PROJECT"

launch_experiment() {
    local NAME="$1"
    local SCRIPT="$2"
    local OUTPUT_SUBDIR="$3"

    echo "=== Creating TPU VM: $NAME ==="
    gcloud compute tpus tpu-vm create "$NAME" \
        --zone="$ZONE" \
        --accelerator-type="$TPU_TYPE" \
        --version="$TPU_VERSION" \
        --project="$PROJECT" \
        --spot 2>&1 || echo "VM $NAME may already exist, continuing."

    local REMOTE_CMD="
set -e
cd ~

echo '--- Installing JAX + deps ---'
pip install -q 'jax[tpu]==0.4.30' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install -q flax==0.8.0 orbax-checkpoint==0.6.4 transformers==4.40.0 wandb==0.17.4 datasets einops gcsfs google-cloud-storage sentencepiece tqdm scipy requests
pip install -q 'setuptools<67'
pip install -q 'mlxu>=0.1.13'
pip install -q git+https://github.com/google-deepmind/optax.git
pip install -q git+https://github.com/google/neural-tangents.git

echo '--- Cloning repo ---'
if [ ! -d full-gauss-newton ]; then git clone $REPO; fi
cd full-gauss-newton

echo '--- Import check ---'
python3 -c \"
import EasyLM.optimizers
import EasyLM.checkpoint
import EasyLM.data
import EasyLM.models.llama.llama_model
print('all imports ok')
\"

echo '--- Launching training in tmux ---'
export WANDB_API_KEY='$WANDB_API_KEY'
export WANDB_ENTITY='$WANDB_ENTITY'
export WANDB_PROJECT='$WANDB_PROJECT'
export OUTPUT_DIR='$GCS_BUCKET/$OUTPUT_SUBDIR'

tmux new-session -d -s train \"cd ~/full-gauss-newton && WANDB_API_KEY='$WANDB_API_KEY' WANDB_ENTITY='$WANDB_ENTITY' WANDB_PROJECT='$WANDB_PROJECT' OUTPUT_DIR='$GCS_BUCKET/$OUTPUT_SUBDIR' bash scripts/$SCRIPT 2>&1 | tee /tmp/${NAME}.log\"
echo 'Training running in tmux session: train'
echo 'Attach with: tmux attach -t train'
"
    echo "=== SSHing into $NAME and setting up ==="
    gcloud compute tpus tpu-vm ssh "$NAME" \
        --zone="$ZONE" \
        --project="$PROJECT" \
        --command="$REMOTE_CMD"

    echo "=== $NAME launched. Monitor at: tmux attach -t train ==="
    echo "    or: tail -f /tmp/${NAME}.log"
}

launch_experiment "gfn-adamw" "run_adamw_tpu.sh" "adamw" &
launch_experiment "gfn-muon"  "run_muon_tpu.sh"  "muon"  &
launch_experiment "gfn-gn"    "run_gn_tpu.sh"    "gn"    &
wait

echo ""
echo "=== All three experiments launched ==="
echo "Monitor logs on each VM:"
echo "  gcloud compute tpus tpu-vm ssh gfn-adamw --zone=$ZONE --command='tail -f /tmp/gfn-adamw.log'"
echo "  gcloud compute tpus tpu-vm ssh gfn-muon  --zone=$ZONE --command='tail -f /tmp/gfn-muon.log'"
echo "  gcloud compute tpus tpu-vm ssh gfn-gn    --zone=$ZONE --command='tail -f /tmp/gfn-gn.log'"
echo ""
echo "Attach to live training on a VM:"
echo "  gcloud compute tpus tpu-vm ssh gfn-adamw --zone=$ZONE --command='tmux attach -t train'"
echo ""
echo "IMPORTANT: Delete VMs when done:"
echo "  gcloud compute tpus tpu-vm delete gfn-adamw gfn-muon gfn-gn --zone=$ZONE --project=$PROJECT --quiet"
