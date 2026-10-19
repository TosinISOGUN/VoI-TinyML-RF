"""Figures 3-6 of the paper, regenerated from the reproduced results."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Validated categorical palette, assigned in fixed order.
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
BLUES = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
})


def _bars(ax, x, h, w, color, labels=True, fmt="{:.1f}", **kw):
    bars = ax.bar(x, h, w, color=color, edgecolor="white", linewidth=1.5, **kw)
    if labels:
        for b, v in zip(bars, h):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, fmt.format(v),
                    ha="center", va="bottom", fontsize=8, color=INK)
    return bars


def fig_models(models: list[dict], path: Path):
    names = [m["model"] for m in models]
    acc = np.array([100 * m["accuracy"] for m in models])
    f1 = np.array([100 * m["macro_f1"] for m in models])
    acc_ci = np.array([[100 * v for v in m["ci"]["accuracy"]] for m in models])
    f1_ci = np.array([[100 * v for v in m["ci"]["macro_f1"]] for m in models])
    x, w = np.arange(len(names)), 0.36
    fig, ax = plt.subplots(figsize=(8, 4.4))
    err = dict(ecolor=MUTED, capsize=3, error_kw={"elinewidth": 1})
    _bars(ax, x - w / 2, acc, w, C1, fmt="{:.2f}%", label="Accuracy",
          yerr=[acc - acc_ci[:, 0], acc_ci[:, 1] - acc], **err)
    _bars(ax, x + w / 2, f1, w, C2, fmt="{:.2f}%", label="Macro-F1",
          yerr=[f1 - f1_ci[:, 0], f1_ci[:, 1] - f1], **err)
    ax.set_xticks(x, [f"{n}\n{m['size_kb']:.2f} KB artifact" for n, m in zip(names, models)])
    ax.set_ylim(60, 100)
    ax.set_ylabel("Test performance (%)")
    ax.set_title("Candidate models: accuracy and macro-F1 (95% bootstrap CI)", loc="left", color=INK)
    ax.legend(loc="upper right", ncol=2)
    fig.savefig(path)
    plt.close(fig)


def fig_confusion(cm, path: Path):
    cm = np.asarray(cm)
    pct = cm / cm.sum(axis=1, keepdims=True)
    labels = ["Class 0\nNormal", "Class 1\nElevated", "Class 2\nHigh severity"]
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(cm, cmap=BLUES)
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            dark = cm[i, j] > cm.max() * 0.55
            ax.text(j, i, f"{cm[i, j]}\n({100 * pct[i, j]:.1f}%)", ha="center", va="center",
                    fontsize=10, color="white" if dark else INK)
    ax.set_xticks(range(3), labels)
    ax.set_yticks(range(3), labels)
    ax.set_xlabel("Predicted environmental state")
    ax.set_ylabel("True environmental state")
    ax.set_title("Selected model on the independent test set", loc="left", color=INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Observations")
    fig.savefig(path)
    plt.close(fig)


def fig_communication(rows: list[dict], path: Path):
    b0 = rows[0]
    names = ["B0\nPeriodic", "B1\nTinyML periodic", "B2\nEvent-triggered", "B3\nProposed"]
    series = [
        ("Packets vs B0", [100 * r["packets"] / b0["packets"] for r in rows], C1),
        ("Modeled RF energy vs B0", [100 * r["energy_mj"] / b0["energy_mj"] for r in rows], C2),
        ("PDR", [100 * r["pdr"] for r in rows], C3),
        ("CEDR", [100 * r["cedr"] for r in rows], C4),
    ]
    x, w = np.arange(len(rows)), 0.2
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for k, (lab, vals, col) in enumerate(series):
        _bars(ax, x + (k - 1.5) * w, np.array(vals), w, col, label=lab)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 112)
    ax.set_ylabel("Relative / reliability value (%)")
    ax.set_title("Communication architectures (RF metrics are modeled, mean of 300 realizations)",
                 loc="left", color=INK)
    ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.14))
    fig.savefig(path)
    plt.close(fig)


def fig_ablation(rows: list[dict], path: Path):
    names = [r["variant"] for r in rows]
    pdr = np.array([100 * r["pdr"] for r in rows])
    cedr = np.array([100 * r["cedr"] for r in rows])
    x, w = np.arange(len(rows)), 0.36
    fig, ax = plt.subplots(figsize=(9, 4.6))
    _bars(ax, x - w / 2, pdr, w, C1, fmt="{:.1f}", label="PDR")
    _bars(ax, x + w / 2, cedr, w, C2, fmt="{:.1f}", label="CEDR")
    ax.set_xticks(x, [f"{n.replace('Without ', 'No ')}\n{r['energy_mj'] / 1000:.2f} J"
                      for n, r in zip(names, rows)], fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Reliability (%)")
    ax.set_title("Ablation on a fixed RF-link trace (link seed 202, ML seed 42)", loc="left", color=INK)
    ax.set_xlabel("Variant and modeled RF energy")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.2))
    fig.savefig(path)
    plt.close(fig)
