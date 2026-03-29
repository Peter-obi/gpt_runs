#!/bin/bash
# Prime Intellect / RunPod startup script
# Usage: bash setup_and_run.sh <config_name> [checkpoint_dir]
# Example: bash setup_and_run.sh train_gpt2_774m_sgdm_driftpwr_p03_layerwise_relative
#
# Expects:
#   GITHUB_TOKEN env var set (for private repo clone)
#   GDRIVE_REMOTE env var set (rclone remote name, default: gdrive)

set -e

CONFIG=${1:-train_gpt2_774m_sgdm_driftpwr_p03_layerwise_relative}
GDRIVE_REMOTE=${GDRIVE_REMOTE:-gdrive}
REPO="https://${GITHUB_TOKEN}@github.com/Peter-obi/gpt_runs.git"
WORKDIR="/workspace/gpt_runs"

echo "=== Setup: config=$CONFIG ==="

# 1. Clone repo if not present
if [ ! -d "$WORKDIR" ]; then
    git clone "$REPO" "$WORKDIR"
else
    cd "$WORKDIR" && git pull
fi
cd "$WORKDIR"

# 2. Install deps
pip install tiktoken wandb -q

# 3. Install rclone if not present
if ! command -v rclone &> /dev/null; then
    apt-get update -q && apt-get install -y unzip -q
    curl https://rclone.org/install.sh | bash
fi

# 4. Copy data if not present
mkdir -p data/openwebtext
if [ ! -f data/openwebtext/train.bin ]; then
    echo "=== Copying train.bin from GDrive ==="
    rclone copy ${GDRIVE_REMOTE}:openwebtext/train.bin data/openwebtext/
fi
if [ ! -f data/openwebtext/val.bin ]; then
    echo "=== Copying val.bin from GDrive ==="
    rclone copy ${GDRIVE_REMOTE}:openwebtext/val.bin data/openwebtext/
fi

# 5. Restore checkpoint if exists on GDrive
OUT_DIR="out-$(echo $CONFIG | sed 's/train_gpt2_//')"
mkdir -p "$OUT_DIR"
echo "=== Checking for checkpoint on GDrive: ${GDRIVE_REMOTE}:${OUT_DIR} ==="
rclone copy ${GDRIVE_REMOTE}:${OUT_DIR}/ckpt.pt ${OUT_DIR}/ 2>/dev/null && echo "Checkpoint restored" || echo "No checkpoint found, starting fresh"

# 6. Run training
echo "=== Starting training: $CONFIG ==="
nohup python train_drift_adaptive.py \
    config/train_gpt2.py \
    config/${CONFIG}.py \
    > run.log 2>&1 &

PID=$!
echo "Training running with PID $PID"
echo "Tail logs: tail -f $WORKDIR/run.log"

# 7. Watch for checkpoint changes and sync to GDrive immediately on update
echo "=== Starting checkpoint watcher ==="
LAST_SYNC=0
while kill -0 $PID 2>/dev/null; do
    if [ -f "${OUT_DIR}/ckpt.pt" ]; then
        MTIME=$(stat -c %Y "${OUT_DIR}/ckpt.pt" 2>/dev/null || echo 0)
        if [ "$MTIME" -gt "$LAST_SYNC" ]; then
            rclone copy ${OUT_DIR}/ckpt.pt ${GDRIVE_REMOTE}:${OUT_DIR}/ 2>/dev/null
            LAST_SYNC=$MTIME
            echo "Checkpoint synced to GDrive at $(date)"
        fi
    fi
    sleep 30  # check every 30 seconds
done
# Final sync on exit
rclone copy ${OUT_DIR}/ckpt.pt ${GDRIVE_REMOTE}:${OUT_DIR}/ 2>/dev/null
echo "Training finished. Final checkpoint synced."
