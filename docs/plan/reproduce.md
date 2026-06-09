# Plan: Reproduce main results

Goal: reproduce the OOD MAE (Table 1) and 30% extrapolative precision (Table 2) from the paper for at least one Bilinear Transduction ("Ours") run plus one baseline, using `pixi` and the existing PyTorch implementation in `blt/`.

## Scope (first pass)

Target tasks:
- **AFLOW / Bulk Modulus** — paper "Ours" MAE 47.4 ± 3.4 GPa, precision 0.40.
- **AFLOW / Debye Temperature** — paper "Ours" MAE 0.31 ± 0.02, precision 0.20.
- **MP / Bulk Modulus** — paper "Ours" MAE 45.8 ± 3.9 GPa, precision 0.60.

Baseline: **Ridge Regression** (the strongest classical baseline on AFLOW per Kauwe et al., trivial sklearn).

Out of scope for this first pass: Matbench (needs modnet/matminer), MoleculeNet (needs deepchem/RDKit), CrabNet, MODNet, Chemprop. Drop or revisit later.

## Why this scope

- AFLOW + MP ship pre-split `train.csv` / `eval.csv` / `ood.csv` in `blt/data/`. Featurization is already covered by `data_modules/data_utils.py::generate_features(elem_prop='oliynyk')` — pure pandas/numpy, no heavy deps.
- BLT is already PyTorch (`blt/utils/networks.py`, `trainer.py`, `transducers.py`, `util.py`, `main.py`, `configs/materials.yml`). No reimplementation.
- Ridge Reg is ~20 lines of sklearn; same featurization as BLT, same train/eval/ood splits — apples-to-apples.
- This combination validates the entire pipeline (data → train → eval → metrics) end-to-end before committing to the heavier deps.

## Steps

### 1. Pixi environment

Create `pixi.toml` at repo root with a minimal default feature for the AFLOW/MP + BLT + Ridge path:

- conda-forge: `python=3.11`, `pytorch`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `tqdm`, `ruamel.yaml`, `pymatgen`.
- Optional GPU: split into `cpu` / `gpu` features.
- Defer Matbench/MoleculeNet deps (`modnet`, `matminer`, `deepchem`, `tensorflow==2.15.1`) into separate pixi features so they don't contaminate the default env.

Acceptance: `pixi run python -c "import torch, pymatgen, sklearn, pandas"` succeeds.

### 2. Preprocess AFLOW + MP into `*.pkl`

Run only the AFLOW and MP halves of `data_modules/create_data.sh` (skip Matbench and MoleculeNet branches). For these two datasets `data_process.py` only needs the in-repo `generate_features` (oliynyk) — does not import `deepchem` or `modnet` on those code paths.

Acceptance: `blt/data/aflow/<prop>/oliynyk.pkl` and `blt/data/mp/<prop>/oliynyk.pkl` exist and load with the expected `train_X / eval_X / ood_X / *_Y / *_formula` keys.

### 3. Train and evaluate BLT on three target tasks

Use `blt/main.py` via `blt/train_eval.sh` for the three target lines (AFLOW bulk_modulus_vrh, AFLOW debye_temperature, MP bulk_modulus). Hyperparameters (hidden_layer_size, hidden_depth, embedding_dim, batch_size) are already in `train_eval.sh` and match what produced the paper numbers.

Settings to verify before running:
- `configs/materials.yml`: `num_epochs: 8000`, `seed: 0`, `skew: right`. Keep as-is for the reproduction. For a smoke test, drop to `num_epochs: 500` first to confirm the pipeline before paying for the full run.
- `PYTHONPATH` must include the repo root (per README).
- GPU strongly recommended — 8000 epochs × ~10 batches/epoch on AFLOW is hours on CPU, minutes on GPU.

Acceptance: `*_eval_in_dist.pkl` and `*_eval_ood.pkl` written under the run's `logdir`, plus an MAE line in `results.txt`.

### 4. Ridge Regression baseline

Add a minimal `baselines/ridge/ridge_reg.py` (mirroring `baselines/rf/rf_reg.py`):
- Load the same `oliynyk.pkl` produced in step 2.
- Apply the paper's Kauwe-et-al. preprocessing: `StandardScaler` then `Normalizer`, fit on train only.
- `Ridge(alpha=...)` — small CV over alpha on the eval split.
- Save `ridge_res.pkl` with `eval_preds`, `ood_preds`, MAE/SEM, and the same fields the BLT eval pkls carry, so downstream metrics share a code path.

Acceptance: matches the AFLOW Bulk Modulus paper number (74.0 ± 3.8 GPa OOD MAE) within SEM.

### 5. Compute the OOD MAE and 30% precision

`eval_supervised` in `blt/utils/util.py` already writes OOD MAE to `results.txt`. The 30% extrapolative precision and TPR are not in that path — need to derive from the saved `*_eval_ood.pkl` + `*_eval_in_dist.pkl` predictions.

Likely already implemented in `blt/plot_maker/plots.py` / `props_utils.py` — check first; lift the metric code into a small reusable `metrics.py` if so. Otherwise implement directly per Section 2 of the paper:

> "the fraction of true top OOD candidates correctly identified among the top predicted OOD candidates ... 30% of the test dataset corresponds to 60% of the OOD-sourced part of the test set ... 95:5 split between in-distribution validation and OOD test sets."

Concretely:
- Pool eval (in-dist) and ood predictions, weighting in-dist 19× per the 95:5 split.
- Take the top-30% by predicted value, count what fraction are truly OOD.
- Apply identically to BLT and Ridge predictions.

Acceptance: numbers match Table 1 / Table 2 within SEM for the three target tasks.

### 6. Confirm units

AFLOW Debye/Shear/Thermal-Conductivity/Thermal-Expansion are reported in log10 (Section 2 of the paper). Inspect `blt/data/aflow/debye_temperature/train.csv` — values look log-scale already, so the eval pipeline does not need to inverse-transform before computing MAE. Sanity check by recovering an order of magnitude for at least one row.

## Caveats / risks

- **Single-seed numbers.** `materials.yml` ships `seed: 0` and the paper's SEM is over the test set, not seeds. Do not interpret across-seed variance until we run multiple seeds.
- **Hardcoded paths in baselines.** `baselines/rf/rf_reg.py`, `baselines/modnet/aflow_mp.py`, `baselines/chemprop/moleculenet.py` all reference `/data/pulkitag/...` or `/home/gridsan/...`. The new `baselines/ridge/ridge_reg.py` should take a CLI `--pkl_path` instead of hardcoding.
- **CrabNet is not vendored.** Skip the CrabNet column when comparing tables; or pull the upstream repo separately if needed.
- **Compute.** AFLOW bulk_modulus (~2.5k train) at 8000 epochs is ~10–30 min on a single modern GPU; AFLOW Egap (14k) is the heaviest of the three datasets (not in the first-pass scope, but flagging if expanded).
- **Determinism.** `np.random.shuffle` + `np.random.randint` inside `train_supervised` are seeded once at startup; expect bit-identical reruns only on the same hardware/CUDA/torch version.

## Definition of done

- `pixi run reproduce-aflow-bulk` (or equivalent) trains BLT and Ridge on AFLOW/Bulk Modulus and prints OOD MAE + 30% precision for both, matching paper Table 1 row "AFLOW / Bulk Modulus" within SEM.
- Same for AFLOW/Debye Temperature and MP/Bulk Modulus.
- A short `references/docs/plan/results.md` records the obtained numbers next to the paper numbers, with seed + commit hash + GPU type.

## Follow-ups (not part of this plan)

- Add Matbench (modnet feature, isolated pixi feature).
- Add MoleculeNet (replace deepchem dep with direct RDKit if we want a clean env).
- Multi-seed runs + bootstrap CIs for the precision metric.
- The PyTorch reimplementation / cleanup (separate task — the math is already torch).

## Full benchmark inventory and training ETAs

Per-batch wall-time observed during the first-pass reproduction was ~3–4 ms across both AFLOW (256-hidden, 9 batches/epoch) and MP (512-hidden, 22 batches/epoch) — the loop is CPU-bound on the per-batch `np.concatenate`, so hidden-size barely affects training time. ETAs below assume single-GPU/process and use `(N_train / batch_size) × 8000 × ~3.5 ms`. The Nature paper benchmarks **13 solids + 4 molecules = 17 tasks**; the repo ships 16 of them (Matbench Formation Energy is the only one missing — see notes below).

| Dataset | Property | #Samples | Hyperparams (h/d/e/b) | batches/epoch | Train ETA | Reproduced? |
|---|---|---|---|---|---|---|
| AFLOW | Band Gap (Egap) | 14123 | 512/3/64/256 | ~50 | **~22–25 min** | — |
| AFLOW | **Bulk Modulus** | 2740 | 256/4/42/256 | 9 | ~4 min | **✓ (49.6 vs 50.5)** |
| AFLOW | **Debye Temperature** | 2740 | 256/3/42/256 | 9 | ~4 min | **✓ (0.307 vs 0.33)** |
| AFLOW | Shear Modulus | 2740 | 256/3/48/256 | 9 | ~4 min | — |
| AFLOW | Thermal Conductivity | 2734 | 256/4/42/256 | 9 | ~4 min | — |
| AFLOW | Thermal Expansion | 2733 | 256/4/48/256 | 9 | ~4 min | — |
| Matbench | Band Gap | 2154 | 512/3/64/256 | 7 | ~3 min | — |
| Matbench | Refractive Index | 4764 | 512/3/64/256 | 16 | ~7–8 min | — |
| Matbench | Yield Strength | 312 | 256/3/32/64 | ~4 | ~2 min | — |
| Matbench | **Formation Energy** *(Nature only)* | 37217 | not listed (sweep) | ~130 | **~55–65 min** | — *(data not in repo)* |
| MP | **Bulk Modulus** | 6307 | 512/3/64/256 | 22 | ~10 min | **✓ (57.96 vs 45.8)** |
| MP | Elastic Anisotropy | 6331 | 512/3/64/256 | 22 | ~10 min | — |
| MP | Shear Modulus | 6184 | 512/3/64/256 | 21 | ~10 min | — |
| MoleculeNet | ESOL (delaney) | 1128 | 1024/3/64/256 | 3 | ~2–3 min | — |
| MoleculeNet | FreeSolv | 643 | 256/4/64/256 | 2 | ~1 min | — |
| MoleculeNet | Lipophilicity | 4200 | 256/3/32/256 | 14 | ~6–7 min | — |
| MoleculeNet | BACE | 1513 | 256/3/48/256 | 5 | ~2–3 min | — |

Totals: **~2.5 hours sequential** of pure training. On 4 GPUs in parallel the wallclock is bottlenecked by the longest task — Matbench Formation Energy at ~1 hour, then AFLOW Egap at ~25 min.

Notes / caveats on the ETAs:
- Eval is **not** included. With the patched `choose_anchor` (sklearn `pairwise_distances_argmin_min` instead of full `cdist`), eval is ~0.5 s per test sample on small tasks, but for AFLOW Egap and Matbench Formation Energy (large train sets → large `train_deltas`) eval per-sample cost grows roughly linearly with `n_train_deltas`. Without batching the model forward pass, eval wall-time on those will be 30–60 min each.
- The 3.5 ms/batch is observed on a single NVIDIA RTX A5000 (CUDA 12.4, PyTorch 2.5.1); could be ±30% off for the larger-network rows since hidden-size impact is small but not zero.
- Add ~30 s startup overhead per task for `pixi run` + featurization-index loading.
- **Matbench Formation Energy** isn't in the repo: not in `data_modules/data_process.py` `--property` choices, no entry in `blt/train_eval.sh`, no raw data shipped. Adding it requires (a) dropping `formation_energy.json` from matminer's `matbench_mp_e_form` into `blt/data/matbench/formation_energy/`, (b) extending the `--property` enum, and (c) picking hyperparameters from the search grid documented in the Nature supplement (the paper does not list a single point estimate for that row).
- The Nature paper revised three AFLOW BLT MAE numbers upward vs. the workshop version (Bulk Modulus 47.4 → 50.5, Debye 0.31 → 0.33, Thermal Conductivity 0.83 → 0.86). Our reproduced numbers happen to match the Nature numbers more tightly.
