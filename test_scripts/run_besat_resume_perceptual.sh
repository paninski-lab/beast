#!/bin/bash
#SBATCH -A bfsr-delta-gpu
#SBATCH -p gpuA40x4,gpuA100x4,gpuA40x4-preempt,gpuA100x4-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH -t 08:00:00
#SBATCH -J beast_train
#SBATCH -o /work/nvme/bfsr/xdai3/runs/beast_train_%j.out
#SBATCH -e /work/nvme/bfsr/xdai3/runs/beast_train_%j.err


# --- Setup environment ---
source ~/.bashrc
module load ffmpeg
conda activate beast
cd /u/xdai3/beast

# Set multiprocessing temp directory to a more stable location (avoid /tmp cleanup issues)
export TMPDIR="/work/nvme/bfsr/xdai3/tmp/${SLURM_JOB_ID:-$USER}"
mkdir -p "$TMPDIR"
echo "TMPDIR set to: $TMPDIR"

# --- Define paths ---
CONFIG="configs/vit_perceptual.yaml"
DATA="/work/nvme/bfsr/xdai3/raw_data/beast/test_video1"
CHECKPOINT="/work/nvme/bfsr/xdai3/runs/beast_train_13711940/tb_logs/version_0/checkpoints/epoch=244-step=2695-best.ckpt"

# Define unique output directory per job (using Slurm job name + ID)
OUTPUT_DIR="/work/nvme/bfsr/xdai3/runs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

echo "---------------------------------------"
echo "Job name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node(s): $SLURM_NODELIST"
echo "Output directory: $OUTPUT_DIR"
echo "---------------------------------------"

# --- Run BEAST ---
echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting BEAST training..."

if [ -f "$CHECKPOINT" ]; then
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Found checkpoint: $CHECKPOINT"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Resuming BEAST training from checkpoint..."
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] About to call: beast train --config \"$CONFIG\" --data \"$DATA\" --checkpoint \"$CHECKPOINT\" --output \"$OUTPUT_DIR\""
    # Note: --checkpoint argument may not be supported, checking if it causes issues
    beast train --config "$CONFIG" --data "$DATA" --checkpoint "$CHECKPOINT" --output "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/training_output.log"
else
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] No checkpoint found. Starting new training run."
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] CONFIG=$CONFIG  DATA=$DATA  OUTPUT_DIR=$OUTPUT_DIR"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] About to call: beast train --config \"$CONFIG\" --data \"$DATA\" --output \"$OUTPUT_DIR\""
    beast train --config "$CONFIG" --data "$DATA" --output "$OUTPUT_DIR" 2>&1 | tee "$OUTPUT_DIR/training_output.log"
fi

echo "[$(date +'%Y-%m-%d %H:%M:%S')] BEAST training completed."

conda deactivate
