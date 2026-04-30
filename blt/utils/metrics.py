"""OOD evaluation metrics from the Known Unknowns paper.

The held-out test set is composed of in-distribution (eval) and out-of-distribution
(ood) samples in roughly equal parts. By construction every OOD sample's true label
is greater than every in-distribution one (OOD = top 5% of data by target value).

Two summary metrics are reported in the paper:

- True Positive Rate (TPR): fraction of OOD samples whose predicted value is at or
  above the OOD threshold ``min(ood_y_true)`` (Tables 5/A1).

- 30% extrapolative precision (Tables 2 / 4): of the top 30% of the held-out test
  predictions, what fraction are truly top OOD candidates? Two variants are
  computed here:
    * `precision_at_30` — unweighted: |top30_pred ∩ top30_gt| / k.
    * `weighted_precision_at_30` — re-weights in-distribution items 19× to reflect
      the 95:5 deployment split (paper: "this metric re-weights misranked
      in-distribution errors 19-fold").
"""
from __future__ import annotations

from typing import TypedDict

import numpy as np


class OODMetrics(TypedDict):
    n_eval: int
    n_ood: int
    k_top: int
    ood_mae: float
    ood_sem: float
    eval_mae: float
    eval_sem: float
    tpr: float
    precision_at_30: float
    weighted_precision_at_30: float


def _flatten(a: np.ndarray) -> np.ndarray:
    return np.asarray(a).reshape(-1)


def compute_ood_metrics(
    eval_pred: np.ndarray,
    eval_gt: np.ndarray,
    ood_pred: np.ndarray,
    ood_gt: np.ndarray,
    top_frac: float = 0.30,
    id_weight: float = 19.0,
) -> OODMetrics:
    eval_pred = _flatten(eval_pred)
    eval_gt = _flatten(eval_gt)
    ood_pred = _flatten(ood_pred)
    ood_gt = _flatten(ood_gt)

    n_eval, n_ood = len(eval_gt), len(ood_gt)
    k = int(round(top_frac * (n_eval + n_ood)))

    eval_err = np.abs(eval_pred - eval_gt)
    ood_err = np.abs(ood_pred - ood_gt)
    eval_mae = float(eval_err.mean())
    ood_mae = float(ood_err.mean())
    eval_sem = float(eval_err.std(ddof=1) / np.sqrt(n_eval)) if n_eval > 1 else 0.0
    ood_sem = float(ood_err.std(ddof=1) / np.sqrt(n_ood)) if n_ood > 1 else 0.0

    ood_threshold = ood_gt.min()
    tpr = float(np.mean(ood_pred >= ood_threshold)) if n_ood else 0.0

    pooled_pred = np.concatenate([eval_pred, ood_pred])
    pooled_gt = np.concatenate([eval_gt, ood_gt])
    is_ood = np.concatenate([np.zeros(n_eval, bool), np.ones(n_ood, bool)])

    top_pred_idx = np.argpartition(-pooled_pred, k - 1)[:k]
    top_gt_idx = np.argpartition(-pooled_gt, k - 1)[:k]

    top_pred_mask = np.zeros(len(pooled_pred), bool)
    top_pred_mask[top_pred_idx] = True
    top_gt_mask = np.zeros(len(pooled_gt), bool)
    top_gt_mask[top_gt_idx] = True

    tp = int((top_pred_mask & top_gt_mask).sum())
    precision_at_30 = tp / k if k else 0.0

    in_top_pred_is_ood = top_pred_mask & is_ood
    in_top_pred_is_id = top_pred_mask & ~is_ood
    tp_ood_in_truetop = top_pred_mask & top_gt_mask & is_ood
    fp_ood = in_top_pred_is_ood & ~top_gt_mask
    fp_id = in_top_pred_is_id
    weighted_num = float(tp_ood_in_truetop.sum())
    weighted_den = (
        float(tp_ood_in_truetop.sum()) + float(fp_ood.sum()) + id_weight * float(fp_id.sum())
    )
    weighted_precision_at_30 = weighted_num / weighted_den if weighted_den else 0.0

    return OODMetrics(
        n_eval=n_eval,
        n_ood=n_ood,
        k_top=k,
        ood_mae=ood_mae,
        ood_sem=ood_sem,
        eval_mae=eval_mae,
        eval_sem=eval_sem,
        tpr=tpr,
        precision_at_30=precision_at_30,
        weighted_precision_at_30=weighted_precision_at_30,
    )


def format_metrics(m: OODMetrics) -> str:
    return (
        f"n_eval={m['n_eval']}, n_ood={m['n_ood']}, k(top30%)={m['k_top']}\n"
        f"  Eval MAE = {m['eval_mae']:.4f} ± {m['eval_sem']:.4f}\n"
        f"  OOD  MAE = {m['ood_mae']:.4f} ± {m['ood_sem']:.4f}\n"
        f"  TPR (pred >= min OOD)        = {m['tpr']:.3f}\n"
        f"  Precision@30 (unweighted)    = {m['precision_at_30']:.3f}\n"
        f"  Precision@30 (id weight 19x) = {m['weighted_precision_at_30']:.3f}"
    )
