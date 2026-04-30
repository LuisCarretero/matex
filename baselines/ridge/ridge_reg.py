"""Ridge Regression baseline following Kauwe et al. (2020).

Loads the same `*.pkl` produced by `data_modules/data_process.py` for AFLOW/MP,
applies StandardScaler then Normalizer (fit on train only, per Kauwe), and trains
a Ridge regressor with alpha selected from the eval (in-distribution) split.
Saves predictions + MAE/SEM in a pkl that mirrors the BLT eval pkl layout enough
to share the downstream 30%-precision metric.
"""
import argparse
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import scipy.stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import Normalizer, StandardScaler


def load_pkl(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pkl(data: Any, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f)


def kauwe_preprocess(train_X: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    """Fit StandardScaler + Normalizer on train_X only; transform all splits."""
    scaler = StandardScaler().fit(train_X)
    train_s = scaler.transform(train_X)
    normalizer = Normalizer().fit(train_s)
    out = [normalizer.transform(train_s)]
    for X in others:
        out.append(normalizer.transform(scaler.transform(X)))
    return tuple(out)


def select_alpha(train_X: np.ndarray, train_y: np.ndarray, eval_X: np.ndarray, eval_y: np.ndarray,
                 alphas: list[float]) -> tuple[float, dict[float, float]]:
    scores: dict[float, float] = {}
    for a in alphas:
        m = Ridge(alpha=a).fit(train_X, train_y)
        scores[a] = mean_absolute_error(eval_y, m.predict(eval_X))
    best = min(scores, key=scores.__getitem__)
    return best, scores


def run(pkl_path: str, save_dir: str, alphas: list[float]) -> dict[str, Any]:
    data = load_pkl(pkl_path)
    train_X, train_y = data["train_X"], data["train_Y"].ravel()
    eval_X, eval_y = data["eval_X"], data["eval_Y"].ravel()
    ood_X, ood_y = data["ood_X"], data["ood_Y"].ravel()

    train_Xs, eval_Xs, ood_Xs = kauwe_preprocess(train_X, eval_X, ood_X)

    best_alpha, alpha_scores = select_alpha(train_Xs, train_y, eval_Xs, eval_y, alphas)
    model = Ridge(alpha=best_alpha).fit(train_Xs, train_y)

    eval_preds = model.predict(eval_Xs)
    ood_preds = model.predict(ood_Xs)

    eval_mae = float(mean_absolute_error(eval_y, eval_preds))
    ood_mae = float(mean_absolute_error(ood_y, ood_preds))
    eval_sem = float(scipy.stats.sem(np.abs(eval_y - eval_preds)))
    ood_sem = float(scipy.stats.sem(np.abs(ood_y - ood_preds)))

    print(f"alpha scores (eval MAE): {alpha_scores}")
    print(f"best alpha: {best_alpha}")
    print(f"Eval MAE:  {eval_mae:.4f} ± {eval_sem:.4f}")
    print(f"OOD  MAE:  {ood_mae:.4f} ± {ood_sem:.4f}")
    print(f"R2 eval:   {r2_score(eval_y, eval_preds):.3f}")
    print(f"R2 ood:    {r2_score(ood_y, ood_preds):.3f}")

    results = {
        "alpha": best_alpha,
        "alpha_scores": alpha_scores,
        "eval_preds": eval_preds,
        "ood_preds": ood_preds,
        "eval_gt": eval_y,
        "ood_gt": ood_y,
        "eval_formula": data.get("eval_formula"),
        "ood_formula": data.get("ood_formula"),
        "eval_mae": eval_mae,
        "eval_sem": eval_sem,
        "ood_mae": ood_mae,
        "ood_sem": ood_sem,
    }
    save_pkl(results, os.path.join(save_dir, "ridge_res.pkl"))
    print(f"Saved {os.path.join(save_dir, 'ridge_res.pkl')}")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pkl_path", required=True, help="Path to oliynyk.pkl / magpie.pkl from data_process.py")
    p.add_argument("--save_dir", required=True, help="Where to dump ridge_res.pkl")
    p.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0],
        help="Alpha values to grid-search using the eval split",
    )
    args = p.parse_args()
    run(args.pkl_path, args.save_dir, args.alphas)
