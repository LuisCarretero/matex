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
