"""PCA implementasyonunun doğruluk testleri (scikit-learn referans alınarak)."""

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler as SklearnScaler

from iris_pca_aop.data.preprocessing import StandardScaler
from iris_pca_aop.exceptions import PCAError
from iris_pca_aop.pca import PCA


@pytest.fixture(scope="module")
def X():
    return SklearnScaler().fit_transform(load_iris().data)


def _align(reference, ours):
    """Özvektör işaretleri keyfi olduğundan karşılaştırma öncesi hizala."""
    signs = np.sign(np.sum(reference * ours, axis=1))
    return reference * signs[:, None]


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_matches_sklearn(X, k):
    ours = PCA(n_components=k).fit(X)
    ref = SklearnPCA(n_components=k).fit(X)
    np.testing.assert_allclose(ours.explained_variance_, ref.explained_variance_, atol=1e-10)
    np.testing.assert_allclose(ours.explained_variance_ratio_, ref.explained_variance_ratio_, atol=1e-10)
    np.testing.assert_allclose(ours.components_, _align(ref.components_, ours.components_), atol=1e-10)


def test_known_iris_variance_ratios(X):
    ratios = PCA(n_components=None).fit(X).explained_variance_ratio_
    np.testing.assert_allclose(ratios, [0.7296, 0.2285, 0.0367, 0.0052], atol=1e-4)
    assert ratios.sum() == pytest.approx(1.0)


def test_components_are_orthonormal(X):
    W = PCA(n_components=None).fit(X).components_
    np.testing.assert_allclose(W @ W.T, np.eye(4), atol=1e-12)


def test_scores_are_uncorrelated_and_variance_equals_eigenvalue(X):
    pca = PCA(n_components=None).fit(X)
    cov = np.cov(pca.transform(X), rowvar=False)
    np.testing.assert_allclose(cov, np.diag(pca.eigenvalues_), atol=1e-10)


def test_full_inverse_transform_reconstructs_data(X):
    pca = PCA(n_components=None).fit(X)
    np.testing.assert_allclose(pca.inverse_transform(pca.transform(X)), X, atol=1e-10)
    assert pca.reconstruction_error(X) == pytest.approx(0.0, abs=1e-20)


def test_reconstruction_error_equals_dropped_variance(X):
    pca2 = PCA(n_components=2).fit(X)
    full = PCA(n_components=None).fit(X)
    n = X.shape[0]
    dropped = full.eigenvalues_[2:].sum() * (n - 1) / n / X.shape[1]
    assert pca2.reconstruction_error(X) == pytest.approx(dropped)


def test_sign_convention_is_deterministic(X):
    W = PCA(n_components=None).fit(X).components_
    for row in W:
        assert row[np.argmax(np.abs(row))] > 0


def test_components_for_95_percent(X):
    assert PCA(n_components=None).fit(X).n_components_for_variance(0.95) == 2


def test_transform_before_fit_raises():
    with pytest.raises(PCAError):
        PCA(2).transform(np.zeros((3, 4)))


def test_standard_scaler_matches_sklearn():
    data = load_iris().data
    np.testing.assert_allclose(StandardScaler().fit_transform(data), SklearnScaler().fit_transform(data))
    scaler = StandardScaler().fit(data)
    np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(data)), data)
