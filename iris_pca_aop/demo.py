"""Hata yakalama mekanizmalarını göstermek için kontrollü hata senaryoları.

``python main.py --demo-errors`` ile çalıştırılır. Her senaryo, aspect'lerin
farklı bir hata davranışını gösterir (doğrulama, alan hatası, yeniden deneme,
yedek davranış, beklenmeyen hatanın sarmalanması).
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

from .bootstrap import Application
from .exceptions import IrisPCAError

logger = logging.getLogger("iris_pca_aop.demo")


class _FlakySource:
    """İlk ``failures`` çağrıda ``OSError`` fırlatan, sonra gerçek kaynağa devreden sahte kaynak."""

    def __init__(self, delegate: Callable[[], pd.DataFrame], failures: int) -> None:
        self.delegate = delegate
        self.remaining = failures

    def __call__(self) -> pd.DataFrame:
        if self.remaining > 0:
            self.remaining -= 1
            raise OSError("Ağ sürücüsüne geçici olarak erişilemedi")
        return self.delegate()


def run_error_scenarios(app: Application) -> list[dict[str, str]]:
    """Senaryoları çalıştırır; her birinin sonucunu döndürür."""
    woven = app.woven_classes
    X = np.random.default_rng(0).normal(size=(20, 4))
    frame = woven["IrisDataLoader"]().load()

    def bad_n_components() -> None:
        woven["PCA"](n_components=10).fit(X)

    def nan_in_data() -> None:
        broken = frame.copy()
        broken.loc[[3, 7, 42], "petal_width"] = np.nan
        woven["IrisPreprocessor"]().prepare(broken)

    def negative_measurement() -> None:
        broken = frame.copy()
        broken.loc[0, "sepal_length"] = -5.1
        woven["IrisPreprocessor"]().prepare(broken)

    def transform_before_fit() -> None:
        woven["PCA"](n_components=2).transform(X)

    def feature_mismatch() -> None:
        model = woven["PCA"](n_components=2).fit(X)
        model.transform(X[:, :3])

    def transient_io_error() -> None:
        loader = woven["IrisDataLoader"]()
        # Yalnızca özel veri kaynağı metodunu değiştiriyoruz; public load() aspect'leri aynen geçerli.
        loader._load_sklearn = _FlakySource(loader._load_sklearn, failures=2)  # type: ignore[method-assign]
        loaded = loader.load()
        logger.info("Geçici hatadan sonra veri yüklendi: %s satır", len(loaded))

    def plot_failure_fallback() -> None:
        result = app.pipeline.visualizer.plot_correlation_matrix(pd.DataFrame({"x": [1, 2]}))
        logger.info("Grafik çizilemedi ama pipeline devam etti (dönen değer: %r)", result)

    def unexpected_error() -> None:
        woven["StandardScaler"]().fit("bu bir matris değil")

    scenarios: list[tuple[str, Callable[[], None]]] = [
        ("Geçersiz n_components (doğrulama, pre)", bad_n_components),
        ("Eksik (NaN) veri (doğrulama, pre)", nan_in_data),
        ("Negatif ölçüm değeri (doğrulama, pre)", negative_measurement),
        ("Eğitilmemiş modelle dönüşüm (alan hatası)", transform_before_fit),
        ("Öznitelik sayısı uyuşmazlığı (doğrulama, pre)", feature_mismatch),
        ("Geçici I/O hatası (yeniden deneme)", transient_io_error),
        ("Grafik hatası (yedek davranış / fallback)", plot_failure_fallback),
        ("Beklenmeyen hata (PipelineError ile sarmalama)", unexpected_error),
    ]

    results = []
    for title, scenario in scenarios:
        logger.info("=" * 70)
        logger.info("SENARYO: %s", title)
        try:
            scenario()
            outcome = "Kurtarıldı (hata yönetimi sayesinde devam edildi)"
        except IrisPCAError as exc:
            cause = f" <- {type(exc.__cause__).__name__}" if exc.__cause__ else ""
            outcome = f"Yakalandı: {type(exc).__name__}{cause}: {exc}"
        logger.info("SONUÇ: %s", outcome)
        results.append({"senaryo": title, "sonuc": outcome})
    return results
