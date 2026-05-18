# Project Guidelines

## Package Management
Use `pixi run python` to run Python. `.pixi/` and `blt/log/` are symlinks to `$SCRATCH/matex/…` — home quota is 40 GiB, never let either grow on home.

## Code Style
- Use type hints where practical.
- Be concise — this is research code accompanying a published paper.

## Cluster usage (Perlmutter / NERSC)

See `../../misc/setup-info.md` for the full reference (quotas, QoS table, iris accounting, charging modes). Project-specific rules:

- **Login node**: edits, `pixi install`, git, sbatch submission only. **No** training, preprocessing, GPU code, or `pytest` runs — even though `login23` has a stray A100, it's shared and CPU/RAM-throttled.
- **Interactive GPU** (instant, ≤4 h): `srun -A m5068_g -C gpu -q interactive --gpus 1 --ntasks=1 --cpus-per-task=32 -t HH:MM:SS bash -c "…"`. Use this for everything compute-side, including smoketests.
- **sbatch QoS**: jobs <6 h → `-q premium` (2× cost, much shorter queue); ≥6 h → `regular`/`shared`. Full-node QoS charges all 4 GPUs regardless of `--gpus`; use `gpu_shared` for sub-node jobs.
- **Unbuffered Python** under srun/sbatch/tee/pipe: `python -u …` or `PYTHONUNBUFFERED=1`. Otherwise `.out` files look empty for hours.
- **Kill training with SIGINT** only (`Ctrl-C` / `kill -INT`), never SIGTERM/SIGKILL.
- HOME-via-DVS doesn't support `flock` — keep any lock-using state on `$SCRATCH`.

## References
This codebase accompanies the *Known Unknowns* paper. The source LaTeX lives in `references/knownUnknowns/` (`intro.tex`, `sn-article.tex`) — consult it when writing plan notes or working on plans to ground the work in the original paper.
