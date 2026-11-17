"""Temel Bileşenler Analizi (Principal Component Analysis) — NumPy ile sıfırdan.

Algoritma
---------
1. **Merkezleme**: Her öznitelikten ortalaması çıkarılır  ->  ``Xc = X - μ``
2. **Kovaryans matrisi**: ``C = Xcᵀ Xc / (n - 1)``  (p x p, simetrik)
3. **Özdeğer ayrışımı**: ``C v = λ v``. Simetrik matrisler için
   ``numpy.linalg.eigh`` kullanılır (gerçel özdeğer, ortonormal özvektör).
4. **Sıralama**: Özdeğerler büyükten küçüğe sıralanır. En büyük özdeğere ait
   özvektör, verinin en çok yayıldığı (varyansı en büyük) yöndür = PC1.
5. **İşaret sabitleme**: Özvektörlerin işareti matematiksel olarak keyfidir
   (``v`` ve ``-v`` ikisi de çözümdür). Sonuçların tekrarlanabilir olması
   için her bileşende mutlak değeri en büyük yükleme pozitif yapılır.
6. **İzdüşüm**: ``Z = Xc W``  (W = ilk k özvektör, p x k)

Açıklanan varyans oranı: ``λ_i / Σλ``.
"""

from __future__ import annotations

import numpy as np

from ..exceptions import PCAError


class PCA:
    """Kovaryans matrisinin özdeğer ayrışımı ile PCA.

    Parameters
    ----------
    n_components:
        Tutulacak bileşen sayısı. ``None`` ise tüm bileşenler tutulur.

    Attributes (fit sonrasında)
    ---------------------------
    mean_                      : (p,)   öznitelik ortalamaları
    covariance_                : (p, p) kovaryans matrisi
    eigenvalues_               : (p,)   tüm özdeğerler (büyükten küçüğe)
    components_                : (k, p) temel bileşenler (satırlar = özvektörler)
    explained_variance_        : (k,)   bileşenlerin varyansı (= özdeğerler)
    explained_variance_ratio_  : (k,)   açıklanan varyans oranı
    cumulative_variance_ratio_ : (k,)   kümülatif açıklanan varyans oranı
    loadings_                  : (p, k) yükleme matrisi = özvektör * sqrt(özdeğer)
    """

    def __init__(self, n_components: int | None = 2) -> None:
        self.n_components = n_components
        self.n_features_in_: int | None = None
        self.n_samples_: int | None = None
        self.mean_: np.ndarray | None = None
        self.covariance_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None
        self.cumulative_variance_ratio_: np.ndarray | None = None

    # ------------------------------------------------------------------ eğitim
    def fit(self, X: np.ndarray) -> "PCA":
        """PCA modelinin eğitilmesi (kovaryans + özdeğer ayrışımı)"""
        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape
        k = n_features if self.n_components is None else int(self.n_components)

        self.mean_ = X.mean(axis=0)
        centered = X - self.mean_
        self.covariance_ = centered.T @ centered / (n_samples - 1)

        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance_)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.clip(eigenvalues[order], 0.0, None)  # sayısal -1e-17 gibi değerleri sıfırla
        eigenvectors = eigenvectors[:, order]

        # İşaret sabitleme: her sütunda mutlak değeri en büyük eleman pozitif olsun.
        max_idx = np.argmax(np.abs(eigenvectors), axis=0)
        signs = np.sign(eigenvectors[max_idx, np.arange(n_features)])
        signs[signs == 0] = 1.0
        eigenvectors = eigenvectors * signs

        total_variance = eigenvalues.sum()
        self.n_samples_, self.n_features_in_ = n_samples, n_features
        self.eigenvalues_ = eigenvalues
        self.components_ = eigenvectors[:, :k].T
        self.explained_variance_ = eigenvalues[:k]
        self.explained_variance_ratio_ = eigenvalues[:k] / total_variance if total_variance > 0 else np.zeros(k)
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)
        return self

    # ------------------------------------------------------------------ dönüşüm
    def _check_fitted(self) -> None:
        if self.components_ is None or self.mean_ is None:
            raise PCAError("PCA modeli henüz eğitilmedi; önce fit() çağrılmalı")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Verinin temel bileşen uzayına izdüşürülmesi"""
        self._check_fitted()
        return (np.asarray(X, dtype=float) - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """PCA modelinin eğitilip verinin dönüştürülmesi"""
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Bileşen uzayından orijinal uzaya geri dönüş (yeniden yapılandırma)"""
        self._check_fitted()
        return np.asarray(Z, dtype=float) @ self.components_ + self.mean_

    # ------------------------------------------------------------------ analiz
    def reconstruction_error(self, X: np.ndarray) -> float:
        """Yeniden yapılandırma hatasının (MSE) hesaplanması"""
        X = np.asarray(X, dtype=float)
        reconstructed = self.inverse_transform(self.transform(X))
        return float(np.mean((X - reconstructed) ** 2))

    @property
    def loadings_(self) -> np.ndarray:
        """Yükleme matrisi (p x k): orijinal öznitelik ile bileşen arasındaki korelasyon
        (veri standardize edildiğinde)."""
        self._check_fitted()
        return self.components_.T * np.sqrt(self.explained_variance_)

    def n_components_for_variance(self, threshold: float = 0.95) -> int:
        """Verilen varyans oranını açıklamak için gereken minimum bileşen sayısı"""
        self._check_fitted()
        ratios = self.eigenvalues_ / self.eigenvalues_.sum()
        return int(np.searchsorted(np.cumsum(ratios), threshold) + 1)
