"""Aggregate BLT + Ridge results across the three target tasks for the reproduction."""
from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

from blt.utils.metrics import compute_ood_metrics

PAPER_OOD_MAE = {
    ("aflow", "bulk_modulus_vrh"): {"BLT": (47.4, 3.4), "Ridge": (74.0, 3.8)},
    ("aflow", "debye_temperature"): {"BLT": (0.31, 0.02), "Ridge": (0.45, 0.03)},
    ("mp", "bulk_modulus"): {"BLT": (45.8, 3.9), "Ridge": (151.0, 14.0)},
}
PAPER_PRECISION_30 = {
    ("aflow", "bulk_modulus_vrh"): {"BLT": 0.40, "Ridge": 0.22},
    ("aflow", "debye_temperature"): {"BLT": 0.20, "Ridge": 0.19},
    ("mp", "bulk_modulus"): {"BLT": 0.60, "Ridge": 0.22},
}


def latest_subdir(p: Path) -> Path:
    return sorted([d for d in p.iterdir() if d.is_dir()])[-1]


def load(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)


def blt_metrics(logroot: Path, dataset: str, prop: str, run_dir_pattern: str):
    base = logroot / dataset / prop / run_dir_pattern
    if not base.exists():
        return None
    run_dir = latest_subdir(base)
    indist_pkl = run_dir / "bilinear_eval_in_dist.pkl"
    ood_pkl = run_dir / "bilinear_eval_ood.pkl"
    if not indist_pkl.exists() or not ood_pkl.exists():
        return run_dir, None
    in_dist = load(indist_pkl)
    ood = load(ood_pkl)
    return run_dir, compute_ood_metrics(
        eval_pred=in_dist["preds"], eval_gt=in_dist["gt"],
        ood_pred=ood["preds"], ood_gt=ood["gt"],
    )


def ridge_metrics(pkl_path: Path):
    if not pkl_path.exists():
        return None
    r = load(pkl_path)
    return compute_ood_metrics(
        eval_pred=r["eval_preds"], eval_gt=r["eval_gt"],
        ood_pred=r["ood_preds"], ood_gt=r["ood_gt"],
    )


TASKS = [
    {
        "key": ("aflow", "bulk_modulus_vrh"),
        "label": "AFLOW / Bulk Modulus [GPa]",
        "blt_run_dir": "oliynyk_subtraction_bilinear_hsize256_hnum4_esize42_bsize256",
        "ridge_pkl": "baselines/ridge/aflow_bulk_modulus_vrh/ridge_res.pkl",
    },
    {
        "key": ("aflow", "debye_temperature"),
        "label": "AFLOW / Debye Temperature [log10 K]",
        "blt_run_dir": "oliynyk_subtraction_bilinear_hsize256_hnum3_esize42_bsize256",
        "ridge_pkl": "baselines/ridge/aflow_debye_temperature/ridge_res.pkl",
    },
    {
        "key": ("mp", "bulk_modulus"),
        "label": "MP / Bulk Modulus [GPa]",
        "blt_run_dir": "oliynyk_subtraction_bilinear_hsize512_hnum3_esize64_bsize256",
        "ridge_pkl": "baselines/ridge/mp_bulk_modulus/ridge_res.pkl",
    },
]


def main(repo_root: Path) -> None:
    logroot = repo_root / "blt" / "log"
    rows = []
    for t in TASKS:
        dataset, prop = t["key"]
        blt = blt_metrics(logroot, dataset, prop, t["blt_run_dir"])
        ridge = ridge_metrics(repo_root / t["ridge_pkl"])
        rows.append({"task": t, "blt": blt, "ridge": ridge})

    print(f"\n{'='*100}\nReproduction summary (paper numbers in parentheses)\n{'='*100}\n")
    for row in rows:
        t = row["task"]
        key = t["key"]
        paper_mae = PAPER_OOD_MAE[key]
        paper_p30 = PAPER_PRECISION_30[key]
        print(f"## {t['label']}")
        if row["blt"] is not None and row["blt"][1] is not None:
            run_dir, m = row["blt"]
            pm, ps = paper_mae["BLT"]
            print(f"  BLT   OOD MAE = {m['ood_mae']:8.4f} ± {m['ood_sem']:.4f}  (paper {pm} ± {ps})")
            print(f"        Eval MAE = {m['eval_mae']:7.4f} ± {m['eval_sem']:.4f}")
            print(f"        TPR              = {m['tpr']:.3f}")
            print(f"        Precision@30     = {m['precision_at_30']:.3f}  (paper {paper_p30['BLT']})")
            print(f"        Precision@30 (w) = {m['weighted_precision_at_30']:.3f}")
            print(f"        run: {run_dir.relative_to(repo_root)}")
        else:
            run_dir = row["blt"][0] if row["blt"] is not None else None
            print(f"  BLT   (run not finished, dir={run_dir})")
        if row["ridge"] is not None:
            m = row["ridge"]
            pm, ps = paper_mae["Ridge"]
            print(f"  Ridge OOD MAE = {m['ood_mae']:8.4f} ± {m['ood_sem']:.4f}  (paper {pm} ± {ps})")
            print(f"        Eval MAE = {m['eval_mae']:7.4f} ± {m['eval_sem']:.4f}")
            print(f"        Precision@30     = {m['precision_at_30']:.3f}  (paper {paper_p30['Ridge']})")
            print(f"        Precision@30 (w) = {m['weighted_precision_at_30']:.3f}")
        else:
            print("  Ridge (no result)")
        print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=Path(__file__).resolve().parents[2])
    args = p.parse_args()
    main(Path(args.repo_root))
