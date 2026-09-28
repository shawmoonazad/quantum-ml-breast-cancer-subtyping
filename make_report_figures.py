"""Build the two summary figures used in README.md from the saved results.

Run from the repository root after weeks 2 and 3 have been executed:
    python make_report_figures.py

Reads:
    outputs/week2/results_per_fold.csv
    outputs/week3/results_per_fold.csv              (first run, unweighted KD)
    outputs/week3_weightedKD/results_per_fold.csv   (revised run, class-weighted KD)
Writes:
    figures/bandwidth_sweep.png
    figures/distillation_comparison.png
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Work from the folder this script is in, so it runs correctly from any terminal location.
os.chdir(Path(__file__).resolve().parent)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
QUBITS = [4, 6, 8]
OUT = Path("figures")
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
})


def mean_std(df, by):
    return df.groupby(by)["bal_acc"].agg(["mean", "std"])


# Figure 1: bandwidth sweep
w2 = pd.read_csv("outputs/week2/results_per_fold.csv")
if "config" not in w2.columns:
    w2["config"] = np.where(w2["model"] == "QSVM", "QSVM " + w2["kernel"], w2["model"])
q = w2[w2["model"] == "QSVM"].copy()
q["reps"] = q["kernel"].str.extract(r"reps(\d)").astype(int)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
for ax, n in zip(axes, QUBITS):
    for reps, color, marker in [(1, BLUE, "o"), (2, ORANGE, "s")]:
        s = mean_std(q[(q.qubits == n) & (q.reps == reps)], "bandwidth").sort_index()
        ax.errorbar(s.index, s["mean"], yerr=s["std"], color=color, marker=marker,
                    markersize=6, lw=2, capsize=3, label=f"Quantum kernel, reps {reps}")
    for name, style, label in [("SVM_RBF", "--", "RBF-SVM"), ("LogReg_L2", ":", "Logistic regression")]:
        m = w2[(w2.qubits == n) & (w2.config == name)]["bal_acc"].mean()
        ax.axhline(m, color=MUTED, ls=style, lw=1.5, label=f"{label} (same features)")
    ax.set_xscale("log")
    ax.set_xticks(sorted(q["bandwidth"].unique()))
    ax.set_xticklabels([f"{b:g}" for b in sorted(q["bandwidth"].unique())])
    ax.minorticks_off()
    ax.set_title(f"{n} features / qubits", color=INK)
    ax.set_xlabel("Bandwidth (log scale)")
axes[0].set_ylabel("Balanced accuracy")
axes[0].set_ylim(0.3, 0.9)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(OUT / "bandwidth_sweep.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# Figure 2: distillation, both runs, with references
first = pd.read_csv("outputs/week3/results_per_fold.csv")
weighted = pd.read_csv("outputs/week3_weightedKD/results_per_fold.csv")
teacher = first.loc[first.model.str.startswith("Teacher"), "bal_acc"].mean()

bars = [
    ("VQC, hard labels only", weighted[weighted.model == "VQC-CE"], BLUE, None),
    ("VQC + distillation, first run", first[first.model == "VQC-KD"], ORANGE, None),
    ("VQC + distillation, class-weighted", weighted[weighted.model == "VQC-KD"], AQUA, "///"),
]
fig, ax = plt.subplots(figsize=(8, 3.8))
x = np.arange(len(QUBITS))
width = 0.26
for j, (label, df, color, hatch) in enumerate(bars):
    s = mean_std(df, "qubits").loc[QUBITS]
    ax.bar(x + (j - 1) * width, s["mean"], width - 0.02, yerr=s["std"], capsize=3,
           color=color, hatch=hatch, edgecolor="white", linewidth=0, label=label,
           error_kw={"ecolor": MUTED, "lw": 1})
lr = [w2[(w2.qubits == n) & (w2.config == "LogReg_L2")]["bal_acc"].mean() for n in QUBITS]
ax.scatter(x + 0.43, lr, marker="D", s=40, color=INK, zorder=3,
           label="Logistic regression, same features")
ax.axhline(teacher, color=INK, lw=1.5, label=f"Genome-wide teacher ({teacher:.2f})")
ax.set_xticks(x)
ax.set_xticklabels([f"{n} qubits" for n in QUBITS])
ax.set_ylabel("Balanced accuracy")
ax.set_ylim(0.4, 0.95)
ax.legend(loc="upper center", ncol=2, fontsize=8.5, bbox_to_anchor=(0.5, 1.28))
fig.tight_layout()
fig.savefig(OUT / "distillation_comparison.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved figures/bandwidth_sweep.png and figures/distillation_comparison.png")
