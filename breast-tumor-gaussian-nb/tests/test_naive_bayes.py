"""
Unit test untuk implementasi Naive Bayes tim.

Jalankan:  python -m pytest tests/ -v

Test "kontrak" mengecek bentuk output dan konsistensi matematis — ini
harus lolos apa pun pilihan desain kalian.

Test "pembanding sklearn" hanya relevan jika kalian memilih likelihood
Gaussian. Jika memilih pendekatan lain, test itu otomatis di-skip lewat
variabel ENV:  NB_SKIP_SKLEARN=1 python -m pytest tests/
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_utils import load_xy, stratified_split  # noqa: E402
from src.naive_bayes import NaiveBayesClassifier  # noqa: E402


@pytest.fixture(scope="module")
def data():
    X, y, feats = load_xy()
    return stratified_split(X, y, test_size=0.2, seed=0)


@pytest.fixture(scope="module")
def model(data):
    Xtr, _, ytr, _ = data
    try:
        return NaiveBayesClassifier().fit(Xtr, ytr)
    except NotImplementedError:
        pytest.skip("NaiveBayesClassifier belum diimplementasikan.")


# ---------------- Kontrak ---------------------------------------------------
def test_fit_returns_self(data):
    Xtr, _, ytr, _ = data
    m = NaiveBayesClassifier()
    try:
        assert m.fit(Xtr, ytr) is m
    except NotImplementedError:
        pytest.skip("belum diimplementasikan")


def test_attributes(model, data):
    Xtr = data[0]
    assert list(model.classes_) == [0, 1]
    assert np.shape(model.log_prior_) == (2,)
    assert model.n_features_ == Xtr.shape[1]
    assert np.isclose(np.exp(model.log_prior_).sum(), 1.0)


def test_shapes(model, data):
    Xte = data[1]
    assert model.feature_log_likelihoods(Xte).shape == (len(Xte), Xte.shape[1], 2)
    assert model.joint_log_likelihood(Xte).shape == (len(Xte), 2)
    assert model.predict_proba(Xte).shape == (len(Xte), 2)


def test_proba_valid(model, data):
    P = model.predict_proba(data[1])
    assert np.all(np.isfinite(P))
    assert np.all((P >= 0) & (P <= 1))
    assert np.allclose(P.sum(axis=1), 1.0)


def test_joint_equals_prior_plus_features(model, data):
    """joint_log_likelihood harus = log_prior + Σ feature_log_likelihoods."""
    Xte = data[1]
    expected = model.log_prior_[None, :] + model.feature_log_likelihoods(Xte).sum(axis=1)
    assert np.allclose(model.joint_log_likelihood(Xte), expected)


def test_proba_consistent_with_joint(model, data):
    """predict_proba harus = softmax(joint_log_likelihood) (Bayes' rule)."""
    Xte = data[1]
    jll = model.joint_log_likelihood(Xte)
    jll = jll - jll.max(axis=1, keepdims=True)
    expected = np.exp(jll) / np.exp(jll).sum(axis=1, keepdims=True)
    assert np.allclose(model.predict_proba(Xte), expected, atol=1e-8)


def test_no_underflow_on_extreme_input(model, data):
    x = data[1][:1] * 50.0
    assert np.all(np.isfinite(model.predict_proba(x)))


def test_custom_priors_are_respected(data):
    Xtr, _, ytr, _ = data
    try:
        m = NaiveBayesClassifier(priors=(0.9, 0.1)).fit(Xtr, ytr)
    except NotImplementedError:
        pytest.skip("belum diimplementasikan")
    assert np.allclose(np.exp(m.log_prior_), [0.9, 0.1])


def test_better_than_majority_baseline(model, data):
    _, Xte, _, yte = data
    acc = (model.predict(Xte) == yte).mean()
    baseline = max(yte.mean(), 1 - yte.mean())
    assert acc > baseline


# ---------------- Pembanding sklearn (opsional) ------------------------------
@pytest.mark.skipif(os.environ.get("NB_SKIP_SKLEARN") == "1",
                    reason="Tim tidak memakai likelihood Gaussian.")
def test_matches_sklearn_gaussiannb(model, data):
    """
    Jika kalian memakai Gaussian NB standar (MLE mean & varians per kelas),
    hasil harus hampir sama dengan sklearn. sklearn menambahkan
    var_smoothing kecil ke varians; perbedaan kecil wajar.
    """
    sk = pytest.importorskip("sklearn.naive_bayes")
    Xtr, Xte, ytr, _ = data
    ref = sk.GaussianNB().fit(Xtr, ytr)
    agree = (ref.predict(Xte) == model.predict(Xte)).mean()
    assert agree >= 0.98, f"Hanya {agree:.1%} prediksi sama dengan sklearn GaussianNB"
