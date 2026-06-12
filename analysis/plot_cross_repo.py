"""Cross-repo comparison: how do MLP and plain Transduction fare RELATIVE TO BLT, in matex
(composition-Z, single point estimate) vs matex-fm (MLIP-Z, full hparam-sweep distribution)?

For each shared dataset (MP bulk, MP shear, band gap) and split (ID, OOD):
  * anchor = BLT (via its mean MAE within that repo/dataset/split)  -> ratio 1.0
  * ratio = method MAE / BLT-mean MAE; >1 worse than BLT, <1 better than BLT
  * matex   -> ONE point per method (Basis A, seed 0)
  * matex-fm -> a DISTRIBUTION over the UMA seed-0 head-compare sweep
                (frozen pool + all trainable head sizes) x (argmin + rho-ball)

Absolute units differ (matex GPa/eV shipped-split vs matex-fm log10 GPa/meV top-5%), but the
BLT-relative ratio is unitless, so the comparison is valid. Output: analysis/figures/cross_repo_blt_relative.png

    pixi run python analysis/plot_cross_repo.py
"""
from __future__ import annotations
import csv, pickle, re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, NullFormatter

MATEX = Path("/global/u1/l/luisc440/workspace/OOD-BT/matex")
FM = Path("/global/u1/l/luisc440/workspace/OOD-BT/matex-fm")
FMLOG = Path("/pscratch/sd/l/luisc440/matex-fm_data/logs")
FIGDIR = MATEX / "analysis" / "figures"
PAT = re.compile(r"(ID|OOD)\s+MAE\s*=\s*([0-9.]+)")
HCOLOR = {"mlp": "#3182bd", "transduction": "#31a354", "bilinear": "#e6550d"}


def parse_log(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in open(path):
        mm = PAT.search(line)
        if mm:
            out[mm.group(1)] = float(mm.group(2))
    return out


def classify(name: str):
    if "mlp" in name:
        return "mlp"
    if "trans" in name:
        return "transduction"
    if "blt" in name:
        return "bilinear"
    return None


# ---------- matex point estimates (composition-Z, seed 0) ----------
def matex_mae(ds, prop, fn, n, k, m, mt) -> dict:
    hp = f"{fn}_subtraction_{mt}_hsize{n}_hnum{k}_esize{m}_bsize256"
    rd = sorted(d for d in (MATEX / "blt" / "log" / ds / prop / hp).iterdir() if d.is_dir())[-1]
    out = {}
    for tag, p in (("ID", rd / f"{mt}_eval_in_dist.pkl"), ("OOD", rd / f"{mt}_eval_ood.pkl")):
        d = pickle.load(open(p, "rb"))
        err = np.abs(np.asarray(d["preds"]).reshape(-1) - np.asarray(d["gt"]).reshape(-1))
        out[tag] = float(err.mean())
    return out


MATEX_TASKS = {  # display -> matex (ds, prop, fn, n, k, m)
    "Bulk modulus": ("mp", "bulk_modulus", "oliynyk", 512, 3, 64),
    "Shear modulus": ("mp", "shear_modulus", "oliynyk", 512, 3, 64),
    "Band gap": ("matbench", "band_gap", "magpie", 512, 3, 64),
}
matex_pts = {p: {mt: matex_mae(*cfg, mt) for mt in ("mlp", "transduction", "bilinear")}
             for p, cfg in MATEX_TASKS.items()}


# ---------- matex-fm sweep distributions (UMA, seed 0) ----------
def fm_moduli(task: str) -> dict:  # task in {bulk, shear}
    d = FMLOG / "moduli_head_compare_54285655"
    res = {"mlp": [], "transduction": [], "bilinear": []}
    nm = {"mlp": "mlp", "trans": "transduction", "blt": "bilinear"}
    for r in ("mlp", "blt", "trans"):
        for head in ("pool", "nl3"):
            for prefix in ("", "rho_"):
                if prefix and r == "mlp":
                    continue  # MLP not transductive -> no rho-ball
                lg = parse_log(d / f"{prefix}{task}-uma-{r}-{head}-s0.log")
                if "ID" in lg and "OOD" in lg:
                    res[nm[r]].append(lg)
    return res


SANE = 1e4  # meV; drop diverged/NaN-blowup runs (band gaps are <~15 eV)


def _ok(lg, name):
    if not ("ID" in lg and "OOD" in lg):
        return False
    if lg["ID"] > SANE or lg["OOD"] > SANE:          # divergence guard
        return False
    if "blt" in name and "sg1" in name:              # BLT stop-grad: unstable, not a requested dim
        return False
    return True


def fm_gap() -> dict:
    res = {"mlp": [], "transduction": [], "bilinear": []}
    # trainable-head argmin runs (seed 0)
    for hd in ("head_mlp_control_54183841", "head_trans_control_54184325",
               "head_blt_stopgrad_54180626"):
        for f in sorted((FMLOG / hd).glob("gap-uma-*-s0.log")):
            m = classify(f.name)
            lg = parse_log(f)
            if m and _ok(lg, f.name):
                res[m].append(lg)
    # frozen-pool argmin from a6_summary (gap rows; 3-seed mean per config)
    rows: dict = {}
    for row in csv.DictReader(open(FM / "scripts/analysis/outputs/a6_summary.csv")):
        if row["task"] != "gap":
            continue
        rows.setdefault((row["depth"], row["readout"]), {})[row["split"]] = float(row["mae_mean"])
    rmap = {"blt256": "bilinear", "mlp": "mlp", "trans": "transduction"}
    for (_, ro), d in rows.items():
        if ro in rmap and "ID" in d and "OOD" in d:
            res[rmap[ro]].append({"ID": d["ID"], "OOD": d["OOD"]})
    # rho-ball runs (frozen + trainable), seed 0 only
    for f in sorted((FMLOG / "a6_rhoball_eval_54191900").glob("rho_gap-uma-*-s0.log")):
        m = classify(f.name)
        lg = parse_log(f)
        if m and _ok(lg, f.name):
            res[m].append(lg)
    return res


FM_DIST = {"Bulk modulus": fm_moduli("bulk"), "Shear modulus": fm_moduli("shear"),
           "Band gap": fm_gap()}

# ---------- ratios (method MAE / BLT-mean MAE) ----------
print("harvest summary (n points per method; BLT-mean MAE):")
RATIO = {}  # prop -> dist -> {"mlp":[...], "transduction":[...], "matex":{...}, "blt_spread":[...]}
for prop in MATEX_TASKS:
    RATIO[prop] = {}
    fm = FM_DIST[prop]
    print(f"  {prop}: mlp={len(fm['mlp'])} trans={len(fm['transduction'])} blt={len(fm['bilinear'])}")
    for dist in ("ID", "OOD"):
        blt_vals = [r[dist] for r in fm["bilinear"]]
        blt_mean = float(np.mean(blt_vals))
        d = {"blt_mean": blt_mean,
             "mlp": [r[dist] / blt_mean for r in fm["mlp"]],
             "transduction": [r[dist] / blt_mean for r in fm["transduction"]],
             "blt_spread": [v / blt_mean for v in blt_vals]}
        mx = matex_pts[prop]
        d["matex"] = {"mlp": mx["mlp"][dist] / mx["bilinear"][dist],
                      "transduction": mx["transduction"][dist] / mx["bilinear"][dist]}
        RATIO[prop][dist] = d


# ---------- plot: 2 panels (ID, OOD), 3 dataset groups, MLP|Trans per group ----------
props = list(MATEX_TASKS)
methods = ["mlp", "transduction"]
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
fig.subplots_adjust(left=0.08, right=0.99, top=0.91, bottom=0.13, hspace=0.1)
group_w = 1.0
off = {"mlp": -0.2, "transduction": 0.2}
for ax, dist in zip(axes, ["ID", "OOD"]):
    for gi, prop in enumerate(props):
        d = RATIO[prop][dist]
        # BLT anchor spread band (min-max of BLT/BLT-mean across the sweep) — faint, behind
        lo, hi = min(d["blt_spread"]), max(d["blt_spread"])
        ax.add_patch(plt.Rectangle((gi - 0.42, lo), group_w - 0.16, hi - lo,
                                   color=HCOLOR["bilinear"], alpha=0.06, zorder=0,
                                   edgecolor="none"))
        for mt in methods:
            x = gi + off[mt]
            ys = d[mt]
            # matex-fm sweep: jittered strip + mean tick
            jit = (np.arange(len(ys)) - (len(ys) - 1) / 2) * 0.03
            ax.scatter([x + j for j in jit], ys, s=42, color=HCOLOR[mt], alpha=0.75,
                       edgecolors="none", zorder=3)
            if ys:
                ax.plot([x - 0.13, x + 0.13], [np.mean(ys)] * 2, color=HCOLOR[mt], lw=2.6, zorder=4)
            # matex point: black-edged star
            ax.scatter([x], [d["matex"][mt]], s=250, marker="*", color=HCOLOR[mt],
                       edgecolors="black", linewidths=1.3, zorder=5)
    ax.axhline(1.0, color=HCOLOR["bilinear"], lw=1.6, ls="--", zorder=2)
    ax.set_xticks(range(len(props)))
    ax.set_xticklabels(props, fontsize=11)
    ax.set_xlim(-0.6, len(props) - 0.4)
    # log y, zoomed to the data
    vlist = [v for prop in props for v in (RATIO[prop][dist]["mlp"]
             + RATIO[prop][dist]["transduction"] + list(RATIO[prop][dist]["matex"].values())
             + RATIO[prop][dist]["blt_spread"])]
    vmin, vmax = min(vlist), max(vlist)
    ax.set_yscale("log")
    ax.set_ylim(vmin * 0.94, vmax * 1.06)
    ticks = [t for t in (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0)
             if vmin * 0.9 <= t <= vmax * 1.1]
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_yticklabels([f"{t:g}" for t in ticks], fontsize=8)
    ax.set_ylabel("MAE ÷ BLT-mean MAE  (log)", fontsize=10)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.text(0.006, 0.96, dist, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")
    msg = "below line = beats BLT" if dist == "ID" else "above line = BLT beats them"
    ax.text(0.994, 0.96, f"BLT anchor (- -); {msg}", transform=ax.transAxes,
            color="#7a3a12", fontsize=9, ha="right", va="top", style="italic")
leg = [Line2D([], [], marker="o", ls="none", mfc="#3182bd", mec="none", ms=8, label="MLP"),
       Line2D([], [], marker="o", ls="none", mfc="#31a354", mec="none", ms=8, label="Transduction"),
       Line2D([], [], marker="o", ls="none", mfc="#888", mec="none", ms=8, label="matex-fm sweep point (UMA, s0)"),
       Line2D([], [], marker="*", ls="none", mfc="#888", mec="black", ms=14, label="matex point estimate"),
       Line2D([], [], color="#888", lw=2.4, label="matex-fm sweep mean"),
       Line2D([], [], color=HCOLOR["bilinear"], lw=1.6, ls="--", label="BLT anchor (=1)")]
axes[0].legend(handles=leg, fontsize=8.5, loc="upper left", ncol=2, framealpha=0.95,
               bbox_to_anchor=(0.08, 0.99))
fig.suptitle("MLP & Transduction relative to BLT — matex (point) vs matex-fm sweep (distribution), "
             "by dataset & split", fontsize=12.5)
FIGDIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIGDIR / "cross_repo_blt_relative.png", dpi=150)
plt.close(fig)
print("wrote", FIGDIR / "cross_repo_blt_relative.png")
