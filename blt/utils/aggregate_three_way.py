"""Aggregate the native 3-way head comparison (MLP vs Transduction vs BLT) across the 8 tasks.

Reads each run's `{model_type}_eval_{in_dist,ood}.pkl` and prints a per-task table of ID/OOD MAE,
TPR, and precision@30 for the three heads. See docs/plan/three_way_native.md.

    pixi run python -m blt.utils.aggregate_three_way
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from blt.utils.metrics import compute_ood_metrics

HEADS = ["mlp", "transduction", "bilinear"]

# label | dataset | prop | data_filename | n | k | m
TASKS = [
    ("Matbench / Band Gap [eV]",        "matbench", "band_gap",          "magpie",  512, 3, 64),
    ("AFLOW / Egap [eV]",               "aflow",    "Egap",              "oliynyk", 512, 3, 64),
    ("AFLOW / Bulk Modulus [GPa]",      "aflow",    "bulk_modulus_vrh",  "oliynyk", 256, 4, 42),
    ("MP / Bulk Modulus [GPa]",         "mp",       "bulk_modulus",      "oliynyk", 512, 3, 64),
    ("AFLOW / Shear Modulus [GPa]",     "aflow",    "shear_modulus_vrh", "oliynyk", 256, 3, 48),
    ("MP / Shear Modulus [GPa]",        "mp",       "shear_modulus",     "oliynyk", 512, 3, 64),
    ("MP / Elastic Anisotropy",         "mp",       "elastic_anisotropy","oliynyk", 512, 3, 64),
    ("AFLOW / Debye Temp [log10 K]",    "aflow",    "debye_temperature", "oliynyk", 256, 3, 42),
]


def latest_run(logroot: Path, dataset, prop, fn, mt, n, k, m) -> Path | None:
    hp = f"{fn}_subtraction_{mt}_hsize{n}_hnum{k}_esize{m}_bsize256"
    base = logroot / dataset / prop / hp
    if not base.exists():
        return None
    dts = sorted(d for d in base.iterdir() if d.is_dir())
    return dts[-1] if dts else None


def head_metrics(run_dir: Path, mt: str):
    idp, oodp = run_dir / f"{mt}_eval_in_dist.pkl", run_dir / f"{mt}_eval_ood.pkl"
    if not (idp.exists() and oodp.exists()):
        return None
    with open(idp, "rb") as f:
        ind = pickle.load(f)
    with open(oodp, "rb") as f:
        ood = pickle.load(f)
    return compute_ood_metrics(
        eval_pred=ind["preds"], eval_gt=ind["gt"],
        ood_pred=ood["preds"], ood_gt=ood["gt"],
    )


def main(repo_root: Path) -> None:
    logroot = repo_root / "blt" / "log"
    print(f"\n{'='*92}\nNative 3-way head comparison (Basis A: matched depth/width, seed 0)\n{'='*92}")
    for label, dataset, prop, fn, n, k, m in TASKS:
        print(f"\n## {label}   ({dataset}/{prop}, n{n}/k{k}/m{m})")
        print(f"  {'head':<13}{'ID MAE':>18}{'OOD MAE':>20}{'TPR':>8}{'Prec@30':>10}")
        for mt in HEADS:
            rd = latest_run(logroot, dataset, prop, fn, mt, n, k, m)
            if rd is None:
                print(f"  {mt:<13}{'(no run dir)':>18}")
                continue
            mtr = head_metrics(rd, mt)
            if mtr is None:
                print(f"  {mt:<13}{'(running/no pkl)':>18}")
                continue
            print(f"  {mt:<13}"
                  f"{mtr['eval_mae']:>10.4f} ± {mtr['eval_sem']:.4f}"
                  f"{mtr['ood_mae']:>11.4f} ± {mtr['ood_sem']:.4f}"
                  f"{mtr['tpr']:>8.3f}{mtr['precision_at_30']:>10.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=Path(__file__).resolve().parents[2])
    args = p.parse_args()
    main(Path(args.repo_root))
