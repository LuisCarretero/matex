#!/bin/bash
#SBATCH -A m5068_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -J blt_bandgap_wandb
#SBATCH -o slurm-%j.out

# Reproduce matbench band_gap. Pass MODEL_TYPE=bilinear (default) or mlp via
# sbatch --export=ALL,MODEL_TYPE=mlp,RUN_NAME=...
set -euo pipefail

REPO=/global/u1/l/luisc440/workspace/OOD-BT/matex
cd "$REPO/blt"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES=0

MODEL_TYPE=${MODEL_TYPE:-bilinear}
RUN_NAME=${RUN_NAME:-bandgap_${MODEL_TYPE}_8000ep_seed0}

echo "[host] $(hostname)  [date] $(date)  [git] $(git -C "$REPO" rev-parse HEAD)"
echo "[model_type] $MODEL_TYPE  [run_name] $RUN_NAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

pixi run --manifest-path "$REPO" python main.py \
    --model_type="$MODEL_TYPE" \
    --dataset_name=matbench \
    --prop_type=band_gap \
    --data_filename=magpie \
    --hidden_layer_size=512 \
    --hidden_depth=3 \
    --embedding_dim=64 \
    --batch_size=256 \
    --seed=0 \
    --eval_every=200 \
    --wandb_mode=online \
    --wandb_project=matex-blt \
    --wandb_name="$RUN_NAME"
