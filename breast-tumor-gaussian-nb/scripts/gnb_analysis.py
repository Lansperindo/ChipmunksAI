"""
Breast Tumor Diagnosis Classification Using Gaussian Naive Bayes
Based on Morphological Features — script analisis lengkap (tim).

Jalankan dari root repo:   python scripts/gnb_analysis.py

Langkah:
  1. Load & inspeksi dataset (EDA)
  2. Split train/test terstratifikasi (split yang sama dengan web app)
  3. Fit Gaussian Naive Bayes buatan tim (src/gnb_manual.py), tanpa sklearn
  4. Evaluasi (accuracy, precision, recall, F1, confusion matrix, ROC)
  5. Cross-check dengan sklearn GaussianNB
  6. 5-fold stratified CV MENGGUNAKAN MODEL TIM
  7. Worked example: satu pasien, lengkap dengan kontribusi tiap fitur
  8. Cek kesesuaian asumsi Gaussian (histogram vs kurva Gaussian hasil fit)

Catatan perbaikan (debug) dari versi awal:
  - read_csv memakai sep=";" (file memakai titik koma)
  - CV sekarang mengevaluasi model tim, bukan sklearn GaussianNB
  - hue_order ditetapkan agar warna kelas tidak tertukar
  - judul plot distribusi diperbaiki (KDE ≠ likelihood Gaussian model)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (accuracy_score, auc, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_curve)
from sklearn.naive_bayes import GaussianNB

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config  # noqa: E402
from src.data_utils import DATA_PATH, stratified_kfold_indices, stratified_split  # noqa: E402
from src.naive_bayes import NaiveBayesClassifier  # noqa: E402

OUT_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid")
ORDER = ["Benign", "Malignant"]
PAL = {"Benign": "#1F3A5F", "Malignant": "#C2185B"}
class_names = ORDER

# ---------------------------------------------------------------------------
# 1. LOAD & INSPECT DATA
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH, sep=";")
print("=" * 70 + "\n1. DATA OVERVIEW\n" + "=" * 70)
print(f"Shape: {df.shape}")
print(f"Missing values total: {df.isnull().sum().sum()}")
print("\nClass distribution:\n", df["diagnosis"].value_counts())

df["label"] = (df["diagnosis"] == "Malignant").astype(int)
feature_cols = [c for c in df.columns if c not in ("diagnosis", "label")]
if config.FEATURES:
    feature_cols = list(config.FEATURES)
X = df[feature_cols].values
y = df["label"].values

plt.figure(figsize=(5, 4))
sns.countplot(x="diagnosis", hue="diagnosis", data=df, order=ORDER, hue_order=ORDER,
              palette=PAL, legend=False)
plt.title("Class Distribution")
plt.tight_layout()
plt.savefig(OUT_DIR / "01_class_distribution.png", dpi=150)
plt.close()

mean_cols = [c for c in df.columns if c.endswith("_mean")]
plt.figure(figsize=(9, 7))
sns.heatmap(df[mean_cols].corr(), annot=True, fmt=".2f", annot_kws={"size": 7},
            cmap="coolwarm", center=0)
plt.title("Correlation Between 'Mean' Morphological Features")
plt.tight_layout()
plt.savefig(OUT_DIR / "02_correlation_heatmap_mean.png", dpi=150)
plt.close()

key_feats = ["radius_mean", "concave points_mean", "texture_mean", "area_mean"]
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, feat in zip(axes.flat, key_feats):
    sns.kdeplot(data=df, x=feat, hue="diagnosis", hue_order=ORDER, fill=True, ax=ax,
                palette=PAL, common_norm=False)
    ax.set_title(feat)
fig.suptitle("Per-Class Empirical Feature Distributions (KDE)")
plt.tight_layout()
plt.savefig(OUT_DIR / "03_feature_distributions_kde.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 2. TRAIN/TEST SPLIT (same split as the web app & evaluate.py)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = stratified_split(X, y, config.TEST_SIZE, config.RANDOM_SEED)
print("\n" + "=" * 70 + "\n2. TRAIN/TEST SPLIT\n" + "=" * 70)
print(f"Train: {len(y_train)}  |  Test: {len(y_test)}")
print(f"Train class balance: {np.bincount(y_train)}  (0=Benign, 1=Malignant)")
print(f"Test  class balance: {np.bincount(y_test)}")

# ---------------------------------------------------------------------------
# 3. MANUAL GAUSSIAN NAIVE BAYES
# ---------------------------------------------------------------------------
print("\n" + "=" * 70 + "\n3. MANUAL GAUSSIAN NAIVE BAYES — LEARNED PARAMETERS\n" + "=" * 70)
manual_model = NaiveBayesClassifier(priors=config.PRIORS).fit(X_train, y_train)
print(f"Prior P(Benign)    = {manual_model.priors_[0]:.4f}")
print(f"Prior P(Malignant) = {manual_model.priors_[1]:.4f}")
ri = feature_cols.index("radius_mean") if "radius_mean" in feature_cols else 0
print(f"\nExample — learned mean & variance for '{feature_cols[ri]}':")
print(f"  Benign:    mean={manual_model.mean_[0, ri]:.3f}, var={manual_model.var_[0, ri]:.3f}")
print(f"  Malignant: mean={manual_model.mean_[1, ri]:.3f}, var={manual_model.var_[1, ri]:.3f}")

params = pd.DataFrame({
    "feature": feature_cols,
    "mean_benign": manual_model.mean_[0], "var_benign": manual_model.var_[0],
    "mean_malignant": manual_model.mean_[1], "var_malignant": manual_model.var_[1],
})
params.to_csv(RES_DIR / "learned_parameters.csv", index=False)

y_proba_manual = manual_model.predict_proba(X_test)[:, 1]
y_pred_manual = (y_proba_manual >= config.THRESHOLD).astype(int)

# ---------------------------------------------------------------------------
# 4. EVALUATION
# ---------------------------------------------------------------------------
def evaluate(y_true, y_pred, name):
    print(f"\n--- {name} ---")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}  <-- sensitivity for Malignant")
    print(f"F1-score : {f1_score(y_true, y_pred):.4f}")
    print(classification_report(y_true, y_pred, target_names=class_names))
    return confusion_matrix(y_true, y_pred)

print("\n" + "=" * 70 + f"\n4. EVALUATION (threshold = {config.THRESHOLD})\n" + "=" * 70)
cm_manual = evaluate(y_test, y_pred_manual, "Manual Gaussian Naive Bayes")

plt.figure(figsize=(5, 4))
sns.heatmap(cm_manual, annot=True, fmt="d", cmap="Blues", annot_kws={"size": 16},
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Manual Gaussian NB")
plt.tight_layout()
plt.savefig(OUT_DIR / "04_confusion_matrix_manual.png", dpi=150)
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_proba_manual)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", color="#C2185B", lw=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Manual Gaussian NB")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(OUT_DIR / "05_roc_curve.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 5. CROSS-CHECK WITH SKLEARN
# ---------------------------------------------------------------------------
print("\n" + "=" * 70 + "\n5. CROSS-CHECK: sklearn GaussianNB\n" + "=" * 70)
sk_model = GaussianNB(priors=config.PRIORS).fit(X_train, y_train)
p_sk = sk_model.predict_proba(X_test)[:, 1]
agreement = ((p_sk >= config.THRESHOLD).astype(int) == y_pred_manual).mean()
max_diff = np.abs(p_sk - y_proba_manual).max()
print(f"Prediction agreement: {agreement * 100:.2f}%")
print(f"Max |P_manual - P_sklearn|: {max_diff:.2e}")

# ---------------------------------------------------------------------------
# 6. 5-FOLD STRATIFIED CV — TEAM'S MODEL
# ---------------------------------------------------------------------------
print("\n" + "=" * 70 + f"\n6. {config.CV_FOLDS}-FOLD STRATIFIED CV (manual model)\n" + "=" * 70)
rows = []
for k, (tr, te) in enumerate(stratified_kfold_indices(y, config.CV_FOLDS, config.RANDOM_SEED), 1):
    m = NaiveBayesClassifier(priors=config.PRIORS).fit(X[tr], y[tr])
    p = m.predict_proba(X[te])[:, 1]
    yp = (p >= config.THRESHOLD).astype(int)
    f, t, _ = roc_curve(y[te], p)
    rows.append({"fold": k, "accuracy": accuracy_score(y[te], yp),
                 "precision": precision_score(y[te], yp), "recall": recall_score(y[te], yp),
                 "f1": f1_score(y[te], yp), "auc": auc(f, t)})
cv = pd.DataFrame(rows)
print(cv.round(4).to_string(index=False))
print("\nMean ± std:")
print((cv.drop(columns="fold").mean().round(4).astype(str) + " ± "
       + cv.drop(columns="fold").std(ddof=0).round(4).astype(str)).to_string())
cv.to_csv(RES_DIR / "cv_folds.csv", index=False)

# ---------------------------------------------------------------------------
# 7. WORKED EXAMPLE — ONE TEST PATIENT, WITH PER-FEATURE BREAKDOWN
# ---------------------------------------------------------------------------
print("\n" + "=" * 70 + "\n7. WORKED EXAMPLE — ONE TEST PATIENT\n" + "=" * 70)
i = 0
x_ex = X_test[i:i + 1]
lj = manual_model.joint_log_likelihood(x_ex)[0]
pr = manual_model.predict_proba(x_ex)[0]
fll = manual_model.feature_log_likelihoods(x_ex)[0]
contrib = pd.DataFrame({"feature": feature_cols, "value": x_ex[0],
                        "logP(x|Benign)": fll[:, 0], "logP(x|Malignant)": fll[:, 1],
                        "log_ratio(M vs B)": fll[:, 1] - fll[:, 0]})
contrib = contrib.reindex(contrib["log_ratio(M vs B)"].abs().sort_values(ascending=False).index)
print(f"True diagnosis: {class_names[y_test[i]]}")
print(f"log prior ratio log(P(M)/P(B)) = {manual_model.log_prior_[1] - manual_model.log_prior_[0]:.3f}")
print(f"Unnormalized log-score -> Benign: {lj[0]:.3f} | Malignant: {lj[1]:.3f}")
print(f"Posterior -> P(Benign|x)={pr[0]:.4f} | P(Malignant|x)={pr[1]:.4f}")
print("\nTop 8 contributing features:")
print(contrib.head(8).round(3).to_string(index=False))
contrib.to_csv(RES_DIR / "worked_example_breakdown.csv", index=False)

# ---------------------------------------------------------------------------
# 8. GAUSSIAN ASSUMPTION CHECK (histogram vs fitted Gaussian)
# ---------------------------------------------------------------------------
check = [f for f in ["radius_mean", "concave points_mean", "area_se", "concavity_se"] if f in feature_cols]
fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
for ax, feat in zip(axes.flat, check):
    j = feature_cols.index(feat)
    grid = np.linspace(X[:, j].min(), X[:, j].max(), 300)
    for c, name in enumerate(ORDER):
        ax.hist(X_train[y_train == c, j], bins=30, density=True, alpha=0.35, color=PAL[name], label=f"{name} (data)")
        mu, var = manual_model.mean_[c, j], manual_model.var_[c, j]
        ax.plot(grid, np.exp(-0.5 * (grid - mu) ** 2 / var) / np.sqrt(2 * np.pi * var),
                color=PAL[name], lw=2, label=f"{name} (Gaussian fit)")
    ax.set_title(feat)
axes.flat[0].legend(fontsize=8)
fig.suptitle("Empirical distribution vs Gaussian likelihood learned by the model (train set)")
plt.tight_layout()
plt.savefig(OUT_DIR / "06_gaussian_fit_check.png", dpi=150)
plt.close()

extreme = ((y_proba_manual > 0.9999) | (y_proba_manual < 1e-4)).mean()
print(f"\nShare of test posteriors > 0.9999 or < 0.0001: {extreme:.1%}")

# ---------------------------------------------------------------------------
# SUMMARY FILE
# ---------------------------------------------------------------------------
with open(RES_DIR / "results_summary.txt", "w", encoding="utf-8") as f:
    f.write("BREAST TUMOR DIAGNOSIS - GAUSSIAN NAIVE BAYES - RESULTS SUMMARY\n" + "=" * 65 + "\n\n")
    f.write(f"Dataset: {df.shape[0]} samples, {len(feature_cols)} features used\n")
    f.write(f"Class distribution: {df['diagnosis'].value_counts().to_dict()}\n")
    f.write(f"Train/test: {len(y_train)}/{len(y_test)} (stratified, seed {config.RANDOM_SEED})\n")
    f.write(f"Priors: {manual_model.priors_.round(4).tolist()}  |  Threshold: {config.THRESHOLD}\n\n")
    f.write("MANUAL MODEL (hold-out test set):\n")
    f.write(f"  Accuracy : {accuracy_score(y_test, y_pred_manual):.4f}\n")
    f.write(f"  Precision: {precision_score(y_test, y_pred_manual):.4f}\n")
    f.write(f"  Recall   : {recall_score(y_test, y_pred_manual):.4f}\n")
    f.write(f"  F1-score : {f1_score(y_test, y_pred_manual):.4f}\n")
    f.write(f"  ROC AUC  : {roc_auc:.4f}\n")
    tn, fp, fn, tp = cm_manual.ravel()
    f.write(f"  Confusion: TN={tn} FP={fp} FN={fn} TP={tp}\n\n")
    f.write(f"Agreement with sklearn GaussianNB: {agreement*100:.2f}% (max prob diff {max_diff:.1e})\n\n")
    f.write(f"{config.CV_FOLDS}-fold CV (manual model), mean ± std:\n")
    for col in ["accuracy", "precision", "recall", "f1", "auc"]:
        f.write(f"  {col:<9}: {cv[col].mean():.4f} ± {cv[col].std(ddof=0):.4f}\n")
    f.write(f"\nShare of extreme test posteriors (>0.9999 or <0.0001): {extreme:.1%}\n")

print(f"\nAll outputs saved to: {OUT_DIR} and {RES_DIR}")
