"""Ön işleme: temizlik, aykırı değer analizi ve standardizasyon."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..exceptions import PCAError
from .loader import FEATURE_COLUMNS, SPECIES, TARGET_COLUMN


class StandardScaler:
    """Z-skor standardizasyonu: ``z = (x - ortalama) / std``.

    PCA varyansı en büyük yönleri aradığı için ölçeği büyük olan bir öznitelik
    (ör. cm yerine mm) analizi tek başına domine edebilir. Standardizasyon
    tüm öznitelikleri ortalaması 0, standart sapması 1 olacak şekilde
    eşitler. (scikit-learn ile uyumlu olması için ``ddof=0`` kullanılır.)
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Standardizasyon parametrelerinin hesaplanması"""
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        self.scale_ = np.where(std == 0, 1.0, std)  # sabit sütunda sıfıra bölmeyi önle
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Verinin standardize edilmesi"""
        if self.mean_ is None or self.scale_ is None:
            raise PCAError("StandardScaler henüz fit edilmedi")
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Parametrelerin hesaplanıp verinin standardize edilmesi"""
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Standart veriden orijinal ölçeğe dönüş"""
        if self.mean_ is None or self.scale_ is None:
            raise PCAError("StandardScaler henüz fit edilmedi")
        return np.asarray(Z, dtype=float) * self.scale_ + self.mean_


@dataclass
class PreprocessingReport:
    """Ön işleme adımında yapılanların raporu."""

    rows_before: int
    rows_after: int
    dropped_missing: int
    duplicate_rows: int
    outliers: dict[str, int] = field(default_factory=dict)
    scaler_mean: dict[str, float] = field(default_factory=dict)
    scaler_std: dict[str, float] = field(default_factory=dict)


@dataclass
class PreparedData:
    """Ön işleme sonrası PCA'ya hazır veri."""

    X_raw: np.ndarray
    X_scaled: np.ndarray
    y: np.ndarray
    labels: np.ndarray
    feature_names: list[str]
    report: PreprocessingReport


class IrisPreprocessor:
    """Iris verisini PCA'ya hazırlar.

    1. Eksik değer içeren satırları çıkarır (Iris'te yoktur, ama genel kod).
    2. Tekrar eden satırları raporlar (silmez; bkz. ``drop_duplicates``).
    3. IQR (çeyrekler açıklığı) yöntemiyle aykırı değerleri sayar.
    4. Öznitelikleri standardize eder.
    """

    def __init__(
        self,
        drop_duplicates: bool = False,
        iqr_factor: float = 1.5,
        scaler: StandardScaler | None = None,
    ) -> None:
        self.drop_duplicates = drop_duplicates
        self.iqr_factor = iqr_factor
        self.scaler = scaler if scaler is not None else StandardScaler()

    def detect_outliers(self, frame: pd.DataFrame) -> dict[str, int]:
        """IQR yöntemiyle aykırı değerlerin tespiti"""
        result: dict[str, int] = {}
        for col in FEATURE_COLUMNS:
            q1, q3 = frame[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            low, high = q1 - self.iqr_factor * iqr, q3 + self.iqr_factor * iqr
            result[col] = int(((frame[col] < low) | (frame[col] > high)).sum())
        return result

    def prepare(self, frame: pd.DataFrame) -> PreparedData:
        """Veri setinin ön işlenmesi (temizlik + standardizasyon)"""
        rows_before = len(frame)
        clean = frame.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
        dropped_missing = rows_before - len(clean)
        duplicates = int(clean.duplicated().sum())
        if self.drop_duplicates:
            clean = clean.drop_duplicates()
        clean = clean.reset_index(drop=True)

        X_raw = clean[FEATURE_COLUMNS].to_numpy(dtype=float)
        X_scaled = self.scaler.fit_transform(X_raw)
        labels = clean[TARGET_COLUMN].astype(str).to_numpy()
        y = np.array([SPECIES.index(label) for label in labels])

        report = PreprocessingReport(
            rows_before=rows_before,
            rows_after=len(clean),
            dropped_missing=dropped_missing,
            duplicate_rows=duplicates,
            outliers=self.detect_outliers(clean),
            scaler_mean=dict(zip(FEATURE_COLUMNS, np.round(self.scaler.mean_, 4).tolist())),
            scaler_std=dict(zip(FEATURE_COLUMNS, np.round(self.scaler.scale_, 4).tolist())),
        )
        return PreparedData(X_raw, X_scaled, y, labels, list(FEATURE_COLUMNS), report)
