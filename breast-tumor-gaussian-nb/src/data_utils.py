"""
data_utils.py — Pemuatan dan penyiapan dataset WDBC.

Modul ini TIDAK berisi logika Naive Bayes. Isinya hanya:
  * membaca CSV (pemisah ';'),
  * memetakan label diagnosis (M = Malignant, B = Benign) ke angka,
  * metadata fitur untuk UI (nama tampilan, satuan, kelompok),
  * pembagian train/test yang terstratifikasi.

Sumber data:
  W. H. Wolberg, W. N. Street, O. L. Mangasarian, "Breast Cancer Wisconsin
  (Diagnostic)", UCI Machine Learning Repository, 1995.
  https://doi.org/10.24432/C5DW2B
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "wdbc.csv"

# ---- Label -----------------------------------------------------------------
# Kolom 'diagnosis': 'Malignant'/'M' (ganas), 'Benign'/'B' (jinak).
# Konvensi kelas di seluruh proyek: 0 = Benign, 1 = Malignant.
LABEL_MAP = {"B": 0, "M": 1, "BENIGN": 0, "MALIGNANT": 1}
CLASS_NAMES = ["Benign", "Malignant"]
CLASS_NAMES_ID = ["Jinak (Benign)", "Ganas (Malignant)"]

# ---- Metadata fitur ---------------------------------------------------------
BASE_MEASUREMENTS = [
    "radius", "texture", "perimeter", "area", "smoothness",
    "compactness", "concavity", "concave points", "symmetry",
    "fractal_dimension",
]
STAT_SUFFIXES = ["mean", "se", "worst"]

MEASUREMENT_DESC_ID = {
    "radius": "Rata-rata jarak dari pusat ke titik-titik tepi inti sel",
    "texture": "Simpangan baku nilai gray-scale di dalam inti sel",
    "perimeter": "Keliling kontur inti sel",
    "area": "Luas inti sel",
    "smoothness": "Variasi lokal panjang jari-jari (kehalusan tepi)",
    "compactness": "perimeter² / area − 1.0",
    "concavity": "Tingkat keparahan bagian cekung pada kontur",
    "concave points": "Jumlah bagian cekung pada kontur",
    "symmetry": "Simetri inti sel",
    "fractal_dimension": "Pendekatan 'coastline' − 1 (kompleksitas tepi)",
}
# Penjelasan awam (untuk pengguna non-teknis di aplikasi)
MEASUREMENT_PLAIN_ID = {
    "radius": "Seberapa besar inti sel (jari-jari).",
    "texture": "Seberapa tidak rata warna/kecerahan di dalam inti sel.",
    "perimeter": "Panjang garis tepi inti sel.",
    "area": "Luas inti sel.",
    "smoothness": "Seberapa mulus garis tepi inti sel.",
    "compactness": "Seberapa padat bentuknya; bentuk yang berlekuk memberi nilai lebih besar.",
    "concavity": "Seberapa dalam lekukan ke dalam pada garis tepi.",
    "concave points": "Seberapa banyak titik lekukan pada garis tepi.",
    "symmetry": "Seberapa tidak simetris bentuk inti sel.",
    "fractal_dimension": "Seberapa rumit/bergerigi garis tepi inti sel.",
}
STAT_PLAIN_ID = {
    "mean": "rata-rata dari semua inti sel di citra",
    "se": "seberapa bervariasi nilainya antar inti sel (standard error)",
    "worst": "rata-rata dari 3 inti sel paling ekstrem di citra",
}

STAT_DESC_ID = {
    "mean": "rata-rata semua inti pada citra",
    "se": "standard error antar inti pada citra",
    "worst": "rata-rata 3 nilai terbesar pada citra",
}


def all_feature_names() -> list[str]:
    """30 nama kolom fitur dengan urutan seperti di CSV."""
    return [f"{m}_{s}" for s in STAT_SUFFIXES for m in BASE_MEASUREMENTS]


def pretty_name(col: str) -> str:
    """'concave points_worst' -> 'Concave points (worst)'."""
    base, stat = col.rsplit("_", 1)
    return f"{base.replace('_', ' ').capitalize()} ({stat})"


def plain_feature(col: str) -> str:
    base, stat = col.rsplit("_", 1)
    return f"{MEASUREMENT_PLAIN_ID.get(base, base)} Nilai ini adalah {STAT_PLAIN_ID.get(stat, stat)}."


def describe_feature(col: str) -> str:
    base, stat = col.rsplit("_", 1)
    return f"{MEASUREMENT_DESC_ID.get(base, base)} — {STAT_DESC_ID.get(stat, stat)}."


# ---- Loading ----------------------------------------------------------------
def load_dataframe(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Baca CSV mentah. File memakai ';' sebagai pemisah kolom."""
    df = pd.read_csv(path, sep=";")
    df["diagnosis"] = df["diagnosis"].str.strip().str.upper()
    unknown = set(df["diagnosis"]) - set(LABEL_MAP)
    if unknown:
        raise ValueError(f"Label diagnosis tidak dikenal: {unknown}")
    df["label"] = df["diagnosis"].map(LABEL_MAP).astype(int)
    df["diagnosis"] = df["label"].map({0: "B", 1: "M"})   # seragamkan jadi B/M
    if "id" not in df.columns:                             # file tanpa kolom ID
        df.insert(0, "id", np.arange(1, len(df) + 1))
    return df


def load_xy(features: list[str] | None = None, path: Path | str = DATA_PATH):
    """Kembalikan (X, y, feature_names). features=None -> semua 30 fitur."""
    df = load_dataframe(path)
    feats = list(features) if features else all_feature_names()
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise KeyError(f"Fitur tidak ada di dataset: {missing}")
    X = df[feats].to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)
    return X, y, feats


def stratified_split_indices(y, test_size: float = 0.2, seed: int = 42):
    """Indeks (train_idx, test_idx) untuk split terstratifikasi (tanpa sklearn)."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_size))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    return np.array(sorted(train_idx)), np.array(sorted(test_idx))


def stratified_split(X, y, test_size: float = 0.2, seed: int = 42):
    """Split train/test yang menjaga proporsi kelas (tanpa sklearn)."""
    tr, te = stratified_split_indices(y, test_size, seed)
    return X[tr], X[te], y[tr], y[te]


def stratified_kfold_indices(y, k: int = 5, seed: int = 42):
    """Generator (train_idx, test_idx) untuk k-fold terstratifikasi."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        for i, chunk in enumerate(np.array_split(idx, k)):
            folds[i].extend(chunk)
    all_idx = np.arange(len(y))
    for i in range(k):
        test_idx = np.array(sorted(folds[i]))
        train_idx = np.setdiff1d(all_idx, test_idx)
        yield train_idx, test_idx
