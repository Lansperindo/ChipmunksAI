"""
eda.py — Exploratory Data Analysis dataset WDBC.

Jalankan dari root repo:   python scripts/eda.py

Output (figures/):
  class_distribution.png     jumlah Benign vs Malignant
  correlation_heatmap.png    korelasi |r| antar 30 fitur
  feature_distributions.png  histogram 10 fitur "mean" per kelas
  results/eda_summary.json   angka ringkasan (untuk laporan)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data_utils import all_feature_names, load_dataframe, pretty_name  # noqa: E402

FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)
NAVY, ROSE = "#1F3A5F", "#C2185B"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})


def main():
    df = load_dataframe()
    feats = all_feature_names()
    X = df[feats]

    # 1. Distribusi kelas
    counts = df["diagnosis"].value_counts().reindex(["B", "M"])
    fig, ax = plt.subplots(figsize=(3.6, 2.8), dpi=220)
    bars = ax.bar(["Benign", "Malignant"], counts.values, color=[NAVY, ROSE], width=0.55)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 5, f"{v} ({v / len(df):.1%})",
                ha="center", fontsize=8)
    ax.set_ylabel("Jumlah sampel")
    ax.set_title("Distribusi kelas (n = 569)", fontsize=9)
    ax.set_ylim(0, counts.max() * 1.18)
    fig.tight_layout()
    fig.savefig(FIG / "class_distribution.png")
    plt.close(fig)

    # 2. Heatmap korelasi
    corr = X.corr().abs()
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=220)
    im = ax.imshow(corr.values, cmap="RdPu", vmin=0, vmax=1)
    ax.set_xticks(range(len(feats)), [pretty_name(f) for f in feats], rotation=90, fontsize=5)
    ax.set_yticks(range(len(feats)), [pretty_name(f) for f in feats], fontsize=5)
    ax.set_title("Korelasi absolut |r| antar fitur", fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.02)
    fig.tight_layout()
    fig.savefig(FIG / "correlation_heatmap.png")
    plt.close(fig)

    # 3. Distribusi fitur "mean" per kelas
    mean_feats = [f for f in feats if f.endswith("_mean")]
    fig, axes = plt.subplots(2, 5, figsize=(9, 3.6), dpi=220)
    for ax, f in zip(axes.flat, mean_feats):
        b = df.loc[df.label == 0, f]
        m = df.loc[df.label == 1, f]
        bins = np.linspace(X[f].min(), X[f].max(), 30)
        ax.hist(b, bins=bins, color=NAVY, alpha=0.6, label="Benign")
        ax.hist(m, bins=bins, color=ROSE, alpha=0.6, label="Malignant")
        ax.set_title(pretty_name(f), fontsize=7)
        ax.tick_params(labelsize=5)
        ax.set_yticks([])
    axes.flat[0].legend(fontsize=6, frameon=False)
    fig.suptitle("Distribusi 10 fitur 'mean' per kelas", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "feature_distributions.png")
    plt.close(fig)

    # 4. Ringkasan angka
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    pairs = upper.stack().sort_values(ascending=False)
    summary = {
        "n_samples": int(len(df)),
        "n_features": len(feats),
        "class_counts": {"Benign": int(counts["B"]), "Malignant": int(counts["M"])},
        "missing_values": int(X.isna().sum().sum()),
        "duplicate_rows": int(df[all_feature_names()].duplicated().sum()),
        "n_pairs_abs_corr_gt_0.9": int((pairs > 0.9).sum()),
        "top_correlated_pairs": [
            {"a": a, "b": b, "abs_r": round(float(v), 4)} for (a, b), v in pairs.head(10).items()
        ],
        "zero_counts": {k: int(v) for k, v in (X == 0).sum().items() if v > 0},
        "skewness_top5": {k: round(float(v), 3)
                          for k, v in X.skew().sort_values(ascending=False).head(5).items()},
    }
    (RES / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nGambar tersimpan di {FIG}/")


if __name__ == "__main__":
    main()
