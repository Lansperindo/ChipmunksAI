"""
gnb_manual.py — Implementasi Gaussian Naive Bayes dari nol (kode tim).

Isi kelas di bawah ini disalin apa adanya dari script analisis tim.
Tidak ada perubahan pada logika matematisnya.
"""
import numpy as np


class GaussianNaiveBayesManual:
    """
    Implements exactly the formula from the lecture:

        P(C | x_1,...,x_n) = alpha * P(C) * prod_i P(x_i | C)

    where each P(x_i | C) is modeled as a Gaussian N(mu_{i,C}, sigma^2_{i,C})
    estimated from the training data. We work in log-space to avoid
    numerical underflow when multiplying 30 small probabilities together,
    then exponentiate + normalize (the "alpha" step) to recover the
    posterior probabilities.
    """

    def __init__(self, var_smoothing=1e-9):
        # var_smoothing: tiny constant added to variance for numerical
        # stability, in case some feature has ~zero variance in a class.
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        n_features = X.shape[1]
        self.mean_ = np.zeros((len(self.classes_), n_features))
        self.var_ = np.zeros((len(self.classes_), n_features))
        self.priors_ = np.zeros(len(self.classes_))

        # widest variance across the whole dataset -> used to scale the
        # smoothing term, same convention sklearn uses
        epsilon = self.var_smoothing * X.var(axis=0).max()

        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.mean_[idx, :] = X_c.mean(axis=0)
            self.var_[idx, :] = X_c.var(axis=0) + epsilon
            self.priors_[idx] = X_c.shape[0] / X.shape[0]
        return self

    def _log_gaussian_pdf(self, class_idx, X):
        # log N(x; mu, sigma^2) summed over all features (independence
        # assumption -> sum of logs instead of product of probabilities)
        mean = self.mean_[class_idx]
        var = self.var_[class_idx]
        log_prob = -0.5 * np.sum(np.log(2.0 * np.pi * var))
        log_prob = log_prob - 0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
        return log_prob

    def predict_log_joint(self, X):
        # log[ P(C) * prod_i P(x_i|C) ] = log P(C) + sum_i log P(x_i|C)
        joint = np.zeros((X.shape[0], len(self.classes_)))
        for idx in range(len(self.classes_)):
            joint[:, idx] = np.log(self.priors_[idx]) + self._log_gaussian_pdf(idx, X)
        return joint

    def predict_proba(self, X):
        log_joint = self.predict_log_joint(X)
        # normalize with the log-sum-exp trick (this IS the "alpha" from
        # the lecture's normalization step, done safely in log-space)
        max_log = log_joint.max(axis=1, keepdims=True)
        log_sum = max_log + np.log(np.exp(log_joint - max_log).sum(axis=1, keepdims=True))
        log_posterior = log_joint - log_sum
        return np.exp(log_posterior)

    def predict(self, X):
        log_joint = self.predict_log_joint(X)
        return self.classes_[np.argmax(log_joint, axis=1)]
