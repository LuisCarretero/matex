"""Report OOD metrics from a saved BLT logdir or a Ridge baseline pkl.

Usage:
  pixi run python -m blt.utils.report --blt-logdir <logdir>
  pixi run python -m blt.utils.report --ridge-pkl baselines/ridge/.../ridge_res.pkl
"""
from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

from blt.utils.metrics import compute_ood_metrics, format_metrics


def _load(p: str | Path):
    with open(p, "rb") as f:
        return pickle.load(f)


def report_blt(logdir: str, model_type: str = "bilinear") -> None:
    in_dist = _load(os.path.join(logdir, f"{model_type}_eval_in_dist.pkl"))
    ood = _load(os.path.join(logdir, f"{model_type}_eval_ood.pkl"))
    m = compute_ood_metrics(
        eval_pred=in_dist["preds"],
        eval_gt=in_dist["gt"],
        ood_pred=ood["preds"],
        ood_gt=ood["gt"],
    )
    print(f"BLT  ({logdir})")
    print(format_metrics(m))


def report_ridge(pkl_path: str) -> None:
    r = _load(pkl_path)
    m = compute_ood_metrics(
        eval_pred=r["eval_preds"],
        eval_gt=r["eval_gt"],
        ood_pred=r["ood_preds"],
        ood_gt=r["ood_gt"],
    )
    print(f"Ridge ({pkl_path}) [alpha={r.get('alpha')}]")
    print(format_metrics(m))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--blt-logdir", default=None)
    p.add_argument("--ridge-pkl", default=None)
    p.add_argument("--model-type", default="bilinear")
    args = p.parse_args()
    if args.blt_logdir:
        report_blt(args.blt_logdir, args.model_type)
    if args.ridge_pkl:
        report_ridge(args.ridge_pkl)
    if not (args.blt_logdir or args.ridge_pkl):
        p.error("Pass at least one of --blt-logdir or --ridge-pkl")
