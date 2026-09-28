"""
metrics.py — Metrik evaluasi klasifikasi biner (kelas positif = 1 = Malignant).

Modul ini hanya menghitung angka. Metrik MANA yang paling penting untuk
kasus diagnosis tumor, dan threshold berapa yang dipakai, adalah keputusan
tim yang harus dijustifikasi di laporan.
"""
from __future__ import annotations

import numpy as np


def confusion_counts(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def _safe_div(a, b):
    return a / b if b else float("nan")


def classification_report(y_true, y_pred) -> dict:
    c = confusion_counts(y_true, y_pred)
    tp, tn, fp, fn = c["TP"], c["TN"], c["FP"], c["FN"]
    sens = _safe_div(tp, tp + fn)          # recall kelas Malignant
    spec = _safe_div(tn, tn + fp)
    prec = _safe_div(tp, tp + fp)          # PPV
    npv = _safe_div(tn, tn + fn)
    f1 = _safe_div(2 * prec * sens, prec + sens)
    return {
        **c,
        "accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "npv": npv,
        "f1": f1,
        "balanced_accuracy": (sens + spec) / 2,
    }


def roc_curve_points(y_true, scores):
    """Titik (FPR, TPR, threshold) untuk kurva ROC, tanpa sklearn."""
    y_true, scores = np.asarray(y_true), np.asarray(scores, dtype=float)
    thresholds = np.r_[np.inf, np.unique(scores)[::-1]]
    P, N = max(np.sum(y_true == 1), 1), max(np.sum(y_true == 0), 1)
    fpr, tpr = [], []
    for t in thresholds:
        pred = scores >= t
        tpr.append(np.sum(pred & (y_true == 1)) / P)
        fpr.append(np.sum(pred & (y_true == 0)) / N)
    return np.array(fpr), np.array(tpr), thresholds


def auc(fpr, tpr) -> float:
    order = np.argsort(fpr)
    return float(np.trapezoid(np.asarray(tpr)[order], np.asarray(fpr)[order]))
