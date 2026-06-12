"""Plots for the native 3-way head comparison (MLP vs Transduction vs BLT), seed 0, Basis A.

Two deliverables (written to analysis/figures/):
  1. ku_degradation_3way.png  — "Known Unknowns" edition (mirrors
     dataset-research/analysis/known_unknowns/figures/ku_degradation.png) but with OUR three
     heads: OOD/ID MAE blow-up per method across our 8 tasks, ordered by PES-derivability tier.
  2. mirror_{gap,bulk,shear}.png — matex-fm style (mirrors matex-fm scripts/analysis/outputs/
     moduli_*.png / final_results.png): two-panel ID (top) / OOD (bottom) MAE, faceted by source
     suite (each self-scaled with its own unit), 3 method bars each, mean ± per-test SEM.

Self-contained: reads each run's {model_type}_eval_{in_dist,ood}.pkl directly.
    pixi run python analysis/plot_three_way.py
"""
from __future__ import annotations
import pickle
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
LOGROOT = REPO / "blt" / "log"
FIGDIR = REPO / "analysis" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

HEADS = ["mlp", "transduction", "bilinear"]
HLABEL = {"mlp": "MLP", "transduction": "Transduction", "bilinear": "BLT (bilinear)"}
# matex-fm head-comparison palette: MLP blue, Transduction green, BLT orange (BLT = "ours").
HCOLOR = {"mlp": "#3182bd", "transduction": "#31a354", "bilinear": "#e6550d"}


def metrics(ds, prop, fn, n, k, m, mt):
    """MAE±SEM for one (task, head) from results.txt-equivalent eval pkls. None if absent."""
    hp = f"{fn}_subtraction_{mt}_hsize{n}_hnum{k}_esize{m}_bsize256"
    base = LOGROOT / ds / prop / hp
    if not base.exists():
        return None
    dts = sorted(d for d in base.iterdir() if d.is_dir())
    if not dts:
        return None
    rd = dts[-1]
    idp, oodp = rd / f"{mt}_eval_in_dist.pkl", rd / f"{mt}_eval_ood.pkl"
    if not (idp.exists() and oodp.exists()):
        return None
    out = {}
    for tag, p in (("id", idp), ("ood", oodp)):
        d = pickle.load(open(p, "rb"))
        err = np.abs(np.asarray(d["preds"]).reshape(-1) - np.asarray(d["gt"]).reshape(-1))
        out[f"{tag}_mae"] = float(err.mean())
        out[f"{tag}_sem"] = float(err.std(ddof=1) / np.sqrt(err.size))
    return out


# (label, suite, ds, prop, fn, unit, short, tier, n, k, m)
TASKS = [
    ("Band gap (MatBench)", "MatBench", "matbench", "band_gap",          "magpie",  "eV",        "Band gap",  "T3", 512, 3, 64),
    ("Band gap (AFLOW)",    "AFLOW",    "aflow",    "Egap",              "oliynyk", "eV",        "Band gap",  "T3", 512, 3, 64),
    ("Bulk mod. (AFLOW)",   "AFLOW",    "aflow",    "bulk_modulus_vrh",  "oliynyk", "GPa",       "Bulk mod.", "T1", 256, 4, 42),
    ("Bulk mod. (MP)",      "MP",       "mp",       "bulk_modulus",      "oliynyk", "GPa",       "Bulk mod.", "T1", 512, 3, 64),
    ("Shear mod. (AFLOW)",  "AFLOW",    "aflow",    "shear_modulus_vrh", "oliynyk", "log10 GPa", "Shear mod.","T1", 256, 3, 48),
    ("Shear mod. (MP)",     "MP",       "mp",       "shear_modulus",     "oliynyk", "GPa",       "Shear mod.","T1", 512, 3, 64),
    ("El. aniso. (MP)",     "MP",       "mp",       "elastic_anisotropy","oliynyk", "unitless",  "El. aniso.","T1", 512, 3, 64),
    ("Debye T (AFLOW)",     "AFLOW",    "aflow",    "debye_temperature", "oliynyk", "log10 K",   "Debye T",   "T1", 256, 3, 42),
]

DATA = {}
for (label, suite, ds, prop, fn, unit, short, tier, n, k, m) in TASKS:
    DATA[label] = {h: metrics(ds, prop, fn, n, k, m, h) for h in HEADS}

SUITE_COLOR = {"AFLOW": "#4a5568", "MatBench": "#805ad5", "MP": "#2c7a7b"}

# ---------- dataset-research (Segal et al. paper) numbers, for overlay ----------
DSR = Path("/global/u1/l/luisc440/workspace/OOD-BT/dataset-research")
_RIDGE, _BLT_P, _E2E = "ridge-regression-oliynyk", "segal-bilinear-transduction", ("modnet", "crabnet")
PAPER_TID = {  # our task label -> paper OOD task id
    "Bulk mod. (AFLOW)": "aflow-bulk-modulus", "Bulk mod. (MP)": "mp-bulk-modulus",
    "Shear mod. (AFLOW)": "aflow-shear-modulus", "Shear mod. (MP)": "mp-shear-modulus",
    "El. aniso. (MP)": "mp-elastic-anisotropy", "Debye T (AFLOW)": "aflow-debye-temperature",
    "Band gap (MatBench)": "matbench-expt-gap", "Band gap (AFLOW)": "aflow-band-gap",
}


def _load_paper():
    v: dict = {}
    for line in open(DSR / "catalogue" / "results.yaml"):
        if "segal-known-unknowns/" not in line:
            continue
        mm = re.search(r"method:\s*([^,}\s]+)", line)
        mt = re.search(r"task:\s*(segal-known-unknowns/[^,}\s]+)", line)
        mv = re.search(r"value:\s*([0-9.]+)", line)
        if mm and mt and mv:
            v.setdefault(mt.group(1), {})[mm.group(1)] = float(mv.group(1))
    return v


PAPER = _load_paper()


def paper_get(tid, dist, method):  # dist in {"ID","OOD"}
    return PAPER.get("segal-known-unknowns/" + tid + ("-id" if dist == "ID" else ""), {}).get(method)


def paper_e2e(tid, dist):
    vals = [x for x in (paper_get(tid, dist, m) for m in _E2E) if x is not None]
    return min(vals) if vals else None


# ============================================================================
# 1) Known-Unknowns edition — OOD/ID blow-up per method, tasks tier-ordered
# ============================================================================
# order: T1 elastic block then T3 electronic block (the tiers our 8 tasks populate)
order = [t for t in TASKS if t[7] == "T1"] + [t for t in TASKS if t[7] == "T3"]
n_t1 = sum(t[7] == "T1" for t in order)


def bracket(ax, a, b, lab):
    tr = ax.get_xaxis_transform()
    ax.plot([a - 0.42, b + 0.42], [-0.20, -0.20], transform=tr, clip_on=False, color="#4a5568", lw=1.0)
    ax.text((a + b) / 2, -0.23, lab, transform=tr, ha="center", va="top",
            fontsize=10, fontweight="bold", color="#2d3748", clip_on=False)


# paper methods overlaid on the blow-up plot (each has a well-defined ID & OOD)
PMETH = [(_RIDGE, "Ridge", "#a0aec0"), ("modnet", "MODNet", "#9f7aea"),
         ("crabnet", "CrabNet", "#2c7a7b"), (_BLT_P, "BLT", HCOLOR["bilinear"])]
fig, ax = plt.subplots(figsize=(16.5, 7.0))
fig.subplots_adjust(left=0.055, right=0.99, top=0.9, bottom=0.30)
bw = 0.11
xs = np.arange(len(order))
for hi, h in enumerate(HEADS):  # ours: 3 solid bars, left half
    ratios = [DATA[t[0]][h]["ood_mae"] / DATA[t[0]][h]["id_mae"] for t in order]
    ax.bar(xs - 0.31 + hi * bw, ratios, bw, color=HCOLOR[h],
           edgecolor="black" if h == "bilinear" else "none",
           linewidth=1.1 if h == "bilinear" else 0, label="ours · " + HLABEL[h], zorder=3)
for pi, (mk, plab, pc) in enumerate(PMETH):  # paper: 4 hatched bars, right half
    ratios = [(paper_get(PAPER_TID[t[0]], "OOD", mk) or np.nan) /
              (paper_get(PAPER_TID[t[0]], "ID", mk) or np.nan) for t in order]
    ax.bar(xs + 0.05 + pi * bw, ratios, bw, color=pc, hatch="///",
           edgecolor="black" if mk == _BLT_P else "white",
           linewidth=1.1 if mk == _BLT_P else 0.4, label="paper · " + plab, zorder=3)
for i, t in enumerate(order):  # suite swatch + ours/paper subticks
    ax.scatter(i - 0.13, 0.02, s=55, marker="s", color=SUITE_COLOR[t[1]],
               transform=ax.get_xaxis_transform(), clip_on=False, zorder=5)
    ax.text(i - 0.14, -0.015, "ours", transform=ax.get_xaxis_transform(), ha="center",
            va="top", fontsize=6, color="#777")
    ax.text(i + 0.22, -0.015, "paper", transform=ax.get_xaxis_transform(), ha="center",
            va="top", fontsize=6, color="#777")
ax.axhline(1.0, color="black", lw=0.8, ls=":")
ax.set_yscale("log")
ax.set_ylabel("OOD MAE / ID MAE   (log) — higher = worse extrapolation")
ax.set_xticks(xs)
ax.set_xticklabels([t[6] + f"\n{t[1]}" for t in order], fontsize=8.5)
ax.set_xlim(-0.6, len(order) - 0.4)
bracket(ax, 0, n_t1 - 1, "T1 · Elastic (2nd-derivs)")
bracket(ax, n_t1, len(order) - 1, "T3 · Electronic (off-PES)")
ax.set_title("Known Unknowns — OOD-to-ID blow-up by head: ours (native 3-way, solid) vs "
             "dataset-research paper (hatched; ■ = suite)", fontsize=12.5)
ax.legend(fontsize=7.5, loc="upper left", ncol=4)
fig.savefig(FIGDIR / "ku_degradation_3way.png", dpi=150)
plt.close(fig)


# ============================================================================
# 1b) Known-Unknowns edition — predictor-comparison ratio per task, ID | OOD
#     (mirrors ku_bt_relative): reference MAE / BLT MAE, log y; ABOVE 1 = BLT better.
#     references = MLP (filled) and Transduction (hollow), suite-coloured; dumbbell joins them.
# ============================================================================
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, ScalarFormatter, NullFormatter
DX = 0.19           # ours at i-DX, paper at i+DX
YL = {"id": (0.45, 4.0), "ood": (0.9, 3.6)}


def _pt(ax, x, y, c, marker, hollow, top):
    """One marker; if off the top, cap with an up-arrow + value tag."""
    if y is None:
        return
    if y > top:
        ax.scatter(x, top * 0.985, s=70, marker="^", facecolors="white" if hollow else c,
                   edgecolors=c, linewidths=1.5, zorder=4, clip_on=False)
        ax.text(x, top * 1.01, f"{y:.1f}", ha="center", va="bottom", fontsize=6, color=c, zorder=6)
    elif hollow:
        ax.scatter(x, y, s=85, marker=marker, facecolors="white", edgecolors=c, linewidths=1.8, zorder=3)
    else:
        ax.scatter(x, y, s=85, marker=marker, color=c, zorder=3)


fig, axes = plt.subplots(2, 1, figsize=(15.5, 9.2), sharex=True)
fig.subplots_adjust(left=0.07, right=0.985, top=0.92, bottom=0.2, hspace=0.09)
for ax, dist in zip(axes, ["id", "ood"]):
    top = YL[dist][1]
    D = dist.upper()
    ax.axhspan(1.0, top, color="#38a169", alpha=0.08, zorder=0)
    ax.axhline(1.0, color="black", lw=1.1, zorder=2)
    for i, t in enumerate(order):
        lab, c, tid = t[0], SUITE_COLOR[t[1]], PAPER_TID[t[0]]
        # ours (matex): circles at i-DX
        blt = DATA[lab]["bilinear"][f"{dist}_mae"]
        rm = DATA[lab]["mlp"][f"{dist}_mae"] / blt
        rt = DATA[lab]["transduction"][f"{dist}_mae"] / blt
        xo = i - DX
        ax.plot([xo, xo], [min(rm, rt), min(max(rm, rt), top)], color="#cbd5e0", lw=1.5, zorder=1)
        _pt(ax, xo, rm, c, "o", False, top)
        _pt(ax, xo, rt, c, "o", True, top)
        # paper (dataset-research): diamonds at i+DX
        bp = paper_get(tid, D, _BLT_P)
        if bp:
            rr = (paper_get(tid, D, _RIDGE) or 0) / bp or None
            re_ = (paper_e2e(tid, D) or 0) / bp or None
            xp = i + DX
            if rr and re_:
                ax.plot([xp, xp], [min(rr, re_, top), min(max(rr, re_), top)], color="#e6cccc", lw=1.5, zorder=1)
            _pt(ax, xp, rr, c, "D", False, top)
            _pt(ax, xp, re_, c, "D", True, top)
    ax.set_yscale("log")
    ax.set_ylim(*YL[dist])
    ticks = [tk for tk in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5) if YL[dist][0] <= tk <= top]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{tk:g}" for tk in ticks], fontsize=8)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel("reference MAE ÷ BLT MAE  (log)", fontsize=9)
    ax.text(0.006, 0.96, D, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top")
    msg = "below 1 = baseline beats BLT" if dist == "id" else "above 1 = BLT beats baseline"
    ax.text(0.994, 0.96, msg, transform=ax.transAxes, color="#555", fontsize=9,
            style="italic", ha="right", va="top")
    # minor ticks marking the ours|paper sub-positions ("new ticks")
    ax.xaxis.set_minor_locator(FixedLocator([i + s * DX for i in range(len(order)) for s in (-1, 1)]))
    ax.tick_params(axis="x", which="minor", length=4, color="#888")
ax = axes[1]
ax.set_xticks(range(len(order)))
ax.set_xticklabels([t[6] + f"\n{t[1]}" for t in order], fontsize=8.5)
ax.set_xlim(-0.6, len(order) - 0.4)
for i in range(len(order)):  # ours / paper sub-labels under the minor ticks
    ax.text(i - DX, -0.16, "ours", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6, color="#888")
    ax.text(i + DX, -0.16, "paper", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6, color="#888")
bracket(ax, 0, n_t1 - 1, "T1 · Elastic (2nd-derivs)")
bracket(ax, n_t1, len(order) - 1, "T3 · Electronic (off-PES)")
ref_leg = [Line2D([], [], marker="o", ls="none", mfc="#4a5568", mec="#4a5568", ms=9, label="ours · vs MLP"),
           Line2D([], [], marker="o", ls="none", mfc="white", mec="#4a5568", mew=1.9, ms=9, label="ours · vs Transduction"),
           Line2D([], [], marker="D", ls="none", mfc="#4a5568", mec="#4a5568", ms=8, label="paper · vs Ridge"),
           Line2D([], [], marker="D", ls="none", mfc="white", mec="#4a5568", mew=1.9, ms=8, label="paper · vs best-e2e")]
suite_leg = [Patch(color=SUITE_COLOR[s], label=s) for s in SUITE_COLOR]
l1 = axes[0].legend(handles=ref_leg, fontsize=8, loc="upper left", framealpha=0.95, ncol=2)
axes[0].add_artist(l1)
axes[0].legend(handles=suite_leg, fontsize=8, loc="upper right", title="suite", framealpha=0.95)
fig.suptitle("Known Unknowns — baselines relative to BLT: ours (matex native 3-way) vs "
             "dataset-research paper, per task (ID | OOD; above 1 = BLT better)", fontsize=12.5)
fig.savefig(FIGDIR / "ku_relative_3way.png", dpi=150)
plt.close(fig)


# ============================================================================
# 2) matex-fm mirror — per property, ID (top)/OOD (bottom), faceted by source
# ============================================================================
PROPS = {
    "gap":   ("Band gap", [("Band gap (MatBench)", "MatBench", "eV"),
                           ("Band gap (AFLOW)", "AFLOW", "eV")]),
    "bulk":  ("Bulk modulus", [("Bulk mod. (AFLOW)", "AFLOW", "GPa"),
                               ("Bulk mod. (MP)", "MP", "GPa")]),
    "shear": ("Shear modulus", [("Shear mod. (AFLOW)", "AFLOW", "log10 GPa"),
                                ("Shear mod. (MP)", "MP", "GPa")]),
}
for key, (title, sources) in PROPS.items():
    ncol = len(sources)
    fig, axes = plt.subplots(2, ncol, figsize=(4.2 * ncol, 7.2), squeeze=False)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.88, bottom=0.12, hspace=0.28, wspace=0.28)
    for ci, (label, suite, unit) in enumerate(sources):
        for ri, dist in enumerate(["id", "ood"]):
            ax = axes[ri][ci]
            vals = [DATA[label][h][f"{dist}_mae"] for h in HEADS]
            sems = [DATA[label][h][f"{dist}_sem"] for h in HEADS]
            bars = ax.bar(range(len(HEADS)), vals, yerr=sems, capsize=4,
                          color=[HCOLOR[h] for h in HEADS],
                          edgecolor=["black" if h == "bilinear" else "none" for h in HEADS],
                          linewidth=1.2)
            for x, (v, e) in enumerate(zip(vals, sems)):
                ax.text(x, v + e, f"{v:.3g}", ha="center", va="bottom", fontsize=9)
            ax.set_xticks(range(len(HEADS)))
            ax.set_xticklabels([HLABEL[h] for h in HEADS], fontsize=9)
            ax.set_ylim(0, max(v + e for v, e in zip(vals, sems)) * 1.25)
            ax.set_ylabel(f"MAE ({unit})", fontsize=9)
            tag = "In-distribution test MAE" if dist == "id" else "OOD test MAE"
            ax.set_title(f"{suite} — {tag}", fontsize=10)
            ax.grid(axis="y", ls=":", alpha=0.4)
    fig.suptitle(f"{title} — MLP vs Transduction vs BLT  (seed 0, matex composition-Z)",
                 fontsize=12, y=0.965)
    fig.savefig(FIGDIR / f"mirror_{key}.png", dpi=150)
    plt.close(fig)

print("wrote:", *(p.name for p in sorted(FIGDIR.glob("*.png"))))
