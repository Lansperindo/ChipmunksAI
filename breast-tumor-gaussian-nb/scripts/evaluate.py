"""
evaluate.py — Evaluasi model tim dengan hold-out + k-fold CV.

Jalankan dari root repo:   python scripts/evaluate.py

Output:
  results/metrics.json            angka untuk tabel hasil di laporan
  figures/confusion_matrix.png
  figures/roc_curve.png
  figures/threshold_tradeoff.png
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

from src import config  # noqa: E402
from src.data_utils import CLASS_NAMES, load_xy, stratified_kfold_indices, stratified_split  # noqa: E402
from src.metrics import auc, classification_report, roc_curve_points  # noqa: E402
from src.naive_bayes import NaiveBayesClassifier  # noqa: E402

FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

NAVY, ROSE, SLATE = "#1F3A5F", "#C2185B", "#6B7B8C"


def main():
    X, y, feats = load_xy(config.FEATURES)
    print(f"Dataset: {X.shape[0]} sampel, {X.shape[1]} fitur, "
          f"{(y == 1).sum()} Malignant / {(y == 0).sum()} Benign")

    # ---------- Hold-out ----------
    Xtr, Xte, ytr, yte = stratified_split(X, y, config.TEST_SIZE, config.RANDOM_SEED)
    model = NaiveBayesClassifier(priors=config.PRIORS).fit(Xtr, ytr)
    p_te = model.predict_proba(Xte)[:, 1]
    holdout = classification_report(yte, (p_te >= config.THRESHOLD).astype(int))
    fpr, tpr, _ = roc_curve_points(yte, p_te)
    holdout["auc"] = auc(fpr, tpr)

    # ---------- k-fold CV ----------
    cv_rows = []
    for tr, te in stratified_kfold_indices(y, config.CV_FOLDS, config.RANDOM_SEED):
        m = NaiveBayesClassifier(priors=config.PRIORS).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        r = classification_report(y[te], (p >= config.THRESHOLD).astype(int))
        f, t, _ = roc_curve_points(y[te], p)
        r["auc"] = auc(f, t)
        cv_rows.append(r)
    keys = ["accuracy", "sensitivity", "specificity", "precision", "f1", "balanced_accuracy", "auc"]
    cv = {k: {"mean": float(np.mean([r[k] for r in cv_rows])),
              "std": float(np.std([r[k] for r in cv_rows]))} for k in keys}

    out = {
        "features": feats, "priors": config.PRIORS, "threshold": config.THRESHOLD,
        "test_size": config.TEST_SIZE, "seed": config.RANDOM_SEED,
        "holdout": holdout, f"cv_{config.CV_FOLDS}fold": cv,
    }
    (RES / "metrics.json").write_text(json.dumps(out, indent=2))

    print("\n== Hold-out test set ==")
    for k in keys:
        print(f"  {k:<18} {holdout[k]:.4f}")
    print(f"  Confusion: TP={holdout['TP']} FN={holdout['FN']} FP={holdout['FP']} TN={holdout['TN']}")
    print(f"\n== {config.CV_FOLDS}-fold CV (mean ± std) ==")
    for k in keys:
        print(f"  {k:<18} {cv[k]['mean']:.4f} ± {cv[k]['std']:.4f}")

    plot_confusion(holdout)
    plot_roc(fpr, tpr, holdout["auc"])
    plot_threshold(yte, p_te)
    print(f"\nTersimpan: {RES / 'metrics.json'} dan gambar di {FIG}/")


def plot_confusion(r):
    cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8), dpi=200)
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=16,
                color="white" if v > cm.max() / 2 else NAVY, fontweight="bold")
    ax.set_xticks([0, 1], [f"Pred {c}" for c in CLASS_NAMES])
    ax.set_yticks([0, 1], [f"True {c}" for c in CLASS_NAMES])
    ax.set_title("Confusion matrix (hold-out)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrix.png")
    plt.close(fig)


def plot_roc(fpr, tpr, a):
    fig, ax = plt.subplots(figsize=(4.2, 3.8), dpi=200)
    ax.plot(fpr, tpr, color=ROSE, lw=2, label=f"Naive Bayes (AUC = {a:.3f})")
    ax.plot([0, 1], [0, 1], ls="--", color=SLATE, lw=1)
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("ROC curve (hold-out)", fontsize=11)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "roc_curve.png")
    plt.close(fig)


def plot_threshold(y, p):
    ts = np.linspace(0.01, 0.99, 99)
    sens = [classification_report(y, (p >= t).astype(int))["sensitivity"] for t in ts]
    spec = [classification_report(y, (p >= t).astype(int))["specificity"] for t in ts]
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=200)
    ax.plot(ts, sens, color=ROSE, lw=2, label="Sensitivity (Malignant terdeteksi)")
    ax.plot(ts, spec, color=NAVY, lw=2, label="Specificity (Benign terdeteksi)")
    ax.axvline(config.THRESHOLD, color=SLATE, ls="--", lw=1)
    ax.set_xlabel("Threshold P(Malignant | x)")
    ax.set_ylim(0, 1.03)
    ax.set_title("Trade-off threshold (hold-out)", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="lower center")
    fig.tight_layout()
    fig.savefig(FIG / "threshold_tradeoff.png")
    plt.close(fig)


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as e:
        sys.exit(f"[!] {e}\n    Implementasikan src/naive_bayes.py terlebih dahulu.")
