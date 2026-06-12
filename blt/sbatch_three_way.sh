#!/bin/bash
#SBATCH -A m5068_g
#SBATCH -C gpu
#SBATCH -q premium
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH -J three_way_native
#SBATCH -o /pscratch/sd/l/luisc440/matex/blt_log/three_way_native_%j.out

# Native 3-way head comparison: MLP vs Transduction vs BLT (bilinear), Basis A (matched depth k /
# width n; BLT keeps its 2 native trunks + embedding m). 8 tasks x 3 heads - 2 already-native
# (matbench band_gap MLP+BLT) = 22 runs, seed 0, 8000 epochs. See docs/plan/three_way_native.md.
#
#   sbatch blt/sbatch_three_way.sh        # from the repo root
#
# Concurrency-limited queue (<=4 lanes): per-job wall time is very uneven (AFLOW Egap eval is the
# long pole at ~30-60 min/transductive arm), so a fixed-wave barrier would idle lanes. We keep 4
# jobs in flight and launch the next as soon as a lane frees (`wait -n`). The models are tiny and
# CPU-bound (per-batch np.concatenate + single-threaded transductive eval), so round-robin GPU
# assignment with occasional 2-on-a-GPU is harmless. Each run writes a unique logdir
# (dataset/prop/<repr>_<sim>_<model>_h.../<datetime>) so there are no inter-job file collisions.

set -uo pipefail
REPO=/global/u1/l/luisc440/workspace/OOD-BT/matex
cd "$REPO/blt"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO"

LOG="$REPO/blt/log/_three_way_logs/${SLURM_JOB_ID:-manual}"
mkdir -p "$LOG"
echo "[host] $(hostname)  [date] $(date)  [git] $(git -C "$REPO" rev-parse HEAD)"
echo "[logs] $LOG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# name | dataset | property | data_filename | n(hidden) | k(depth) | m(embedding) | model_type
# matbench band_gap: transduction only (MLP+BLT already native). All other tasks: all 3 heads.
# Heavy transductive arms (Egap, then MP) front-loaded so they start on the first free lanes.
JOBS=(
  "aflowEgap-bilinear|aflow|Egap|oliynyk|512|3|64|bilinear"
  "aflowEgap-transduction|aflow|Egap|oliynyk|512|3|64|transduction"
  "aflowEgap-mlp|aflow|Egap|oliynyk|512|3|64|mlp"
  "mpBulk-bilinear|mp|bulk_modulus|oliynyk|512|3|64|bilinear"
  "mpBulk-transduction|mp|bulk_modulus|oliynyk|512|3|64|transduction"
  "mpBulk-mlp|mp|bulk_modulus|oliynyk|512|3|64|mlp"
  "mpShear-bilinear|mp|shear_modulus|oliynyk|512|3|64|bilinear"
  "mpShear-transduction|mp|shear_modulus|oliynyk|512|3|64|transduction"
  "mpShear-mlp|mp|shear_modulus|oliynyk|512|3|64|mlp"
  "mpElAniso-bilinear|mp|elastic_anisotropy|oliynyk|512|3|64|bilinear"
  "mpElAniso-transduction|mp|elastic_anisotropy|oliynyk|512|3|64|transduction"
  "mpElAniso-mlp|mp|elastic_anisotropy|oliynyk|512|3|64|mlp"
  "aflowBulk-bilinear|aflow|bulk_modulus_vrh|oliynyk|256|4|42|bilinear"
  "aflowBulk-transduction|aflow|bulk_modulus_vrh|oliynyk|256|4|42|transduction"
  "aflowBulk-mlp|aflow|bulk_modulus_vrh|oliynyk|256|4|42|mlp"
  "aflowShear-bilinear|aflow|shear_modulus_vrh|oliynyk|256|3|48|bilinear"
  "aflowShear-transduction|aflow|shear_modulus_vrh|oliynyk|256|3|48|transduction"
  "aflowShear-mlp|aflow|shear_modulus_vrh|oliynyk|256|3|48|mlp"
  "aflowDebye-bilinear|aflow|debye_temperature|oliynyk|256|3|42|bilinear"
  "aflowDebye-transduction|aflow|debye_temperature|oliynyk|256|3|42|transduction"
  "aflowDebye-mlp|aflow|debye_temperature|oliynyk|256|3|42|mlp"
  "mbBandgap-transduction|matbench|band_gap|magpie|512|3|64|transduction"
)

run_one() {
  local name=$1 ds=$2 prop=$3 fn=$4 n=$5 k=$6 m=$7 mt=$8 gpu=$9
  echo "[start $name] gpu=$gpu model=$mt $ds/$prop n=$n k=$k m=$m  $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES=$gpu pixi run --manifest-path "$REPO" python main.py \
      --model_type="$mt" \
      --dataset_name="$ds" --prop_type="$prop" --data_filename="$fn" \
      --hidden_layer_size="$n" --hidden_depth="$k" --embedding_dim="$m" \
      --batch_size=256 --seed=0 > "$LOG/$name.log" 2>&1
  echo "[done  $name] rc=$? $(date +%H:%M:%S)"
}

echo "=== launching ${#JOBS[@]} runs, <=4 concurrent ==="
i=0
for job in "${JOBS[@]}"; do
  IFS='|' read -r name ds prop fn n k m mt <<< "$job"
  run_one "$name" "$ds" "$prop" "$fn" "$n" "$k" "$m" "$mt" $((i % 4)) &
  i=$((i + 1))
  while (( $(jobs -r | wc -l) >= 4 )); do wait -n; done
done
wait

echo ""
echo "############### RESULTS (val + ood MAE from results.txt) ###############"
for job in "${JOBS[@]}"; do
  IFS='|' read -r name ds prop fn n k m mt <<< "$job"
  # results.txt lives in the per-run logdir; grab the most recent matching one
  rt=$(find "$REPO/blt/log/$ds/$prop" -type f -name results.txt -path "*_${mt}_*" 2>/dev/null | sort | tail -1)
  printf "## %-28s " "$name"
  if [ -n "$rt" ]; then grep -hE "Type|MAE" "$rt" | tr '\n' ' '; fi
  echo ""
done
echo "THREE-WAY NATIVE COMPLETE job=${SLURM_JOB_ID:-manual} elapsed=${SECONDS}s"
