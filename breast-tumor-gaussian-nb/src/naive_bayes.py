"""
naive_bayes.py — Adapter antarmuka untuk model tim.

Inti algoritma (estimasi prior, mean, varians, log-densitas Gaussian,
normalisasi log-sum-exp) ada di src/gnb_manual.py dan ditulis oleh tim.

Kelas di bawah hanya "lem" agar model tim bisa dipakai web app, script
evaluasi, dan unit test dengan nama method yang seragam:
  * priors      -> opsi menimpa prior hasil estimasi (dari config.py)
  * log_prior_  -> log(priors_)
  * feature_log_likelihoods -> log P(x_i | C) PER fitur (tanpa dijumlah),
                               dipakai UI untuk breakdown kontribusi fitur
  * joint_log_likelihood    -> alias predict_log_joint milik tim
  * predict(threshold)      -> keputusan dengan threshold yang bisa diatur

Unit test tests/test_naive_bayes.py memverifikasi bahwa
  log_prior_ + Σ feature_log_likelihoods == predict_log_joint (kode tim),
jadi breakdown di UI dijamin konsisten dengan perhitungan model tim.
"""
from __future__ import annotations

import numpy as np

from src.gnb_manual import GaussianNaiveBayesManual


class NaiveBayesClassifier(GaussianNaiveBayesManual):
    def __init__(self, priors: tuple[float, float] | None = None, var_smoothing: float = 1e-9):
        super().__init__(var_smoothing=var_smoothing)
        self.priors = priors

    def fit(self, X, y):
        X = np.asarray(X, float)
        super().fit(X, np.asarray(y))
        if self.priors is not None:
            p = np.asarray(self.priors, float)
            if p.shape != (len(self.classes_),) or not np.isclose(p.sum(), 1.0):
                raise ValueError("priors harus berisi 2 angka yang berjumlah 1")
            self.priors_ = p
        self.log_prior_ = np.log(self.priors_)
        self.n_features_ = X.shape[1]
        return self

    def feature_log_likelihoods(self, X):
        """(n_samples, n_features, n_classes): suku-suku yang dijumlahkan
        di _log_gaussian_pdf milik tim, sebelum dijumlahkan."""
        X = np.atleast_2d(np.asarray(X, float))
        mu, var = self.mean_[None, :, :], self.var_[None, :, :]       # (1, C, F)
        ll = -0.5 * np.log(2.0 * np.pi * var) - 0.5 * (X[:, None, :] - mu) ** 2 / var
        return ll.transpose(0, 2, 1)                                   # (n, F, C)

    def joint_log_likelihood(self, X):
        return self.predict_log_joint(np.atleast_2d(np.asarray(X, float)))

    def predict_proba(self, X):
        return super().predict_proba(np.atleast_2d(np.asarray(X, float)))

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
