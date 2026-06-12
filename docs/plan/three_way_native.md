# Plan: native 3-way head comparison (MLP vs Transduction vs BLT)

Drafted 2026-06-11. Goal: benchmark **matex (no-fm)** as a native baseline by comparing three
predictor heads — per-sample **MLP**, plain **Transduction** (`MLP([obs‖δ])`), and **BLT**
(bilinear) — on the original composition-descriptor pipeline, to see how well BLT extrapolates
"natively" against a fair MLP and the unstructured-transduction ablation.

This is matex-only. The sole thing borrowed from `../matex-fm` is the *logic* of the transduction
readout (`TransductionNet = MLP([obs‖δ])`); matex's own pairing + transducer anchor-search are
reused unchanged so BLT stays native.

## Fairness basis: A (matched depth/width)

Per the paper's protocol (`docs/papers/.../appendix_implementation_details.tex:171-174`): all heads
use the same depth `k` and width `n`; BLT carries two trunks + embedding `m`, transduction one
trunk over `[obs‖δ]`, MLP one trunk over `obs`. Fairness = matched `(k, n)`, **not** matched param
count (BLT is ~2× params at matched `(k,n)` — that asymmetry is intrinsic to BLT). Param counts are
reported in the results table for transparency. A param-matched MLP (width ≈ n√2) is deferred.

## Task matrix (8 tasks, seed 0, 8000 epochs)

| # | dataset / property | n/k/m | ~N_train | preprocessed? | native done |
|---|---|---|---|---|---|
| 1 | matbench / `band_gap` | 512/3/64 | 4,144 | ✓ `magpie.pkl` | MLP + BLT |
| 2 | aflow / `Egap` | 512/3/64 | 14,123 | ✗ | — |
| 3 | aflow / `bulk_modulus_vrh` | 256/4/42 | 2,740 | ✗ | BLT (old) |
| 4 | mp / `bulk_modulus` | 512/3/64 | 6,307 | ✗ | BLT (old) |
| 5 | aflow / `shear_modulus_vrh` | 256/3/48 | 2,740 | ✗ | — |
| 6 | mp / `shear_modulus` | 512/3/64 | 6,184 | ✗ | — |
| 7 | mp / `elastic_anisotropy` | 512/3/64 | 6,331 | ✗ | — |
| 8 | aflow / `debye_temperature` | 256/3/42 | 2,740 | ✗ | BLT (old) |

8 tasks × 3 heads = 24; minus 2 already-native (matbench band gap MLP+BLT) = **22 new runs**.

Per-head CLI (task's own `n,k,m`):
- MLP   — `--model_type=mlp         --hidden_layer_size=n --hidden_depth=k`
- Trans — `--model_type=transduction --hidden_layer_size=n --hidden_depth=k`
- BLT   — `--model_type=bilinear    --hidden_layer_size=n --hidden_depth=k --embedding_dim=m`

## Phases

0. **Implement transduction head** (3 files):
   - `blt/utils/networks.py`: `TransductionPredictor` — `mlp(2*input_dim, n, output_dim, k)`,
     `forward(obs, deltas) = trunk(cat([obs, deltas], -1))`, `apply(weight_init)`.
   - `blt/utils/util.py`: register in `define_model`; add `'transduction'` to the anchor-search
     branch (~L82) and the model-forward branch (~L94) of `eval_supervised`.
   - `blt/utils/trainer.py`: extend the `'bilinear' in model_type` forward branch (~L111) to also
     fire for `transduction`. Spectra branch stays bilinear-only.
0.5 **Smoketest** (interactive GPU, never login): `model_type=transduction num_epochs=100` on
   matbench band gap → confirms train + transducer eval + `transduction_eval_*.pkl` + `results.txt`.
1. **Preprocess** the 7 un-preprocessed tasks via `data_modules/data_process.py` (→ `oliynyk.pkl`).
2. **Run** 22 jobs on a 4-GPU `-q premium` node, wave-packed (distinct task per lane). One
   parameterized sbatch over the (task, head) table. Egap gets its own lane (long eval).
3. **Eval + aggregate**: extend `blt/utils/aggregate_results.py` → per-task 3-way table
   (OOD + ID MAE±SEM, TPR, precision@30 from the eval pkls).
4. **Write-up** in `docs/plan/results.md`.

## Compute

Per run = CPU-bound training + single-threaded transductive eval (scales with `n_train_deltas`).
Rough: AFLOW small (debye/bulk/shear) ~30 min/task all 3 arms; MP (bulk/shear/el-aniso) ~70
min/task; matbench band gap +21 min (transduction only). **AFLOW Egap is the long pole** — 14k
train → BLT/Trans eval 30–60 min each. ≈ 7 GPU-hours total → ~2 h wall on 4 GPUs, Egap-bound.
Fits one 4-h premium allocation (split Egap off if it stalls).

## Deferred
- **Formation energy** — not in matex (no data, not in `--property` enum). Separate workstream:
  acquire `matbench_mp_e_form`, featurize (magpie/oliynyk), extend enum, pick hyperparams (paper
  lists none → likely 512/3/64).
- **Param-matched MLP** (width ≈ n√2) — capacity-control robustness arm, +8 runs.
- **Multi-seed** — seed 0 first; 3 seeds for the final table (OOD set deterministic → seeds vary
  only init/pairing/id-subset).
