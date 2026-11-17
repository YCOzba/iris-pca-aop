"""Uçtan uca Iris + PCA iş akışı.

``IrisPCAPipeline`` yalnızca **iş mantığını** içerir: adımları sırayla çağırır
ve sonuçları birleştirir. Loglama, zamanlama, doğrulama, hata yönetimi ve
metrik toplama kodlarının hiçbiri burada yoktur; ``bootstrap.build_pipeline``
bu sınıfı ve bağımlılıklarını aspect'lerle *dokur* (weave).

Her public metodun docstring'inin ilk satırı, ``LoggingAspect`` tarafından
**işlem adımı** olarak loga yazılır.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as SklearnPCA

from .aop.metrics import MetricsRegistry
from .config import PipelineConfig
from .data.loader import DatasetSummary, IrisDataLoader
from .data.preprocessing import IrisPreprocessor, PreparedData
from .pca.pca import PCA
from .visualization.plots import IrisVisualizer


@dataclass
class PCAOutcome:
    """PCA adımının çıktıları."""

    model: PCA
    """Kullanıcının seçtiği ``n_components`` ile eğitilen model."""
    full_model: PCA
    """Tüm bileşenlerle eğitilen model (scree, 3B grafik ve analiz için)."""
    scores: np.ndarray
    full_scores: np.ndarray
    reconstruction_mse: float
    components_for_95: int


@dataclass
class PipelineResult:
    summary: DatasetSummary
    data: PreparedData
    pca: PCAOutcome
    verification: dict[str, Any]
    figures: list[Path] = field(default_factory=list)
    files: dict[str, Path] = field(default_factory=dict)


class IrisPCAPipeline:
    """Iris veri seti üzerinde PCA iş akışı."""

    def __init__(
        self,
        config: PipelineConfig,
        loader: IrisDataLoader,
        preprocessor: IrisPreprocessor,
        pca_factory: Callable[[int | None], PCA],
        visualizer: IrisVisualizer,
        registry: MetricsRegistry | None = None,
    ) -> None:
        self.config = config
        self.loader = loader
        self.preprocessor = preprocessor
        self.pca_factory = pca_factory
        self.visualizer = visualizer
        self.registry = registry

    # ------------------------------------------------------------------ adımlar
    def load_data(self) -> pd.DataFrame:
        """ADIM 1 - Veri setinin yüklenmesi"""
        return self.loader.load()

    def describe_data(self, frame: pd.DataFrame) -> DatasetSummary:
        """ADIM 2 - Veri setinin içeriğinin incelenmesi"""
        return self.loader.describe(frame)

    def preprocess(self, frame: pd.DataFrame) -> PreparedData:
        """ADIM 3 - Ön işleme (temizlik, aykırı değer analizi, standardizasyon)"""
        return self.preprocessor.prepare(frame)

    def apply_pca(self, X: np.ndarray) -> PCAOutcome:
        """ADIM 4 - PCA analizinin uygulanması"""
        model = self.pca_factory(self.config.n_components)
        scores = model.fit_transform(X)
        full_model = self.pca_factory(None)
        full_scores = full_model.fit_transform(X)
        return PCAOutcome(
            model=model,
            full_model=full_model,
            scores=scores,
            full_scores=full_scores,
            reconstruction_mse=model.reconstruction_error(X),
            components_for_95=full_model.n_components_for_variance(0.95),
        )

    def verify_with_sklearn(self, X: np.ndarray, outcome: PCAOutcome) -> dict[str, Any]:
        """ADIM 5 - Sonuçların scikit-learn PCA ile doğrulanması"""
        reference = SklearnPCA(n_components=self.config.n_components).fit(X)
        ref_scores = reference.transform(X)
        # Özvektör işaretleri keyfi olduğu için karşılaştırmadan önce hizalanır.
        signs = np.sign(np.sum(reference.components_ * outcome.model.components_, axis=1))
        ratio_diff = float(np.max(np.abs(reference.explained_variance_ratio_ - outcome.model.explained_variance_ratio_)))
        component_diff = float(np.max(np.abs(reference.components_ * signs[:, None] - outcome.model.components_)))
        score_diff = float(np.max(np.abs(ref_scores * signs - outcome.scores)))
        tolerance = 1e-8
        return {
            "sklearn_explained_variance_ratio": reference.explained_variance_ratio_.round(6).tolist(),
            "bizim_explained_variance_ratio": outcome.model.explained_variance_ratio_.round(6).tolist(),
            "maks_varyans_orani_farki": ratio_diff,
            "maks_bilesen_farki": component_diff,
            "maks_skor_farki": score_diff,
            "tolerans": tolerance,
            "eslesiyor": bool(max(ratio_diff, component_diff, score_diff) < tolerance),
        }

    def visualize(self, frame: pd.DataFrame, data: PreparedData, outcome: PCAOutcome) -> list[Path]:
        """ADIM 6 - Sonuçların görselleştirilmesi"""
        viz = self.visualizer
        names = data.feature_names
        paths = [
            viz.plot_feature_distributions(frame),
            viz.plot_scatter_matrix(frame),
            viz.plot_correlation_matrix(frame),
            viz.plot_scree(outcome.full_model),
            viz.plot_pca_2d(outcome.full_scores, data.labels, outcome.full_model),
            viz.plot_pca_3d(outcome.full_scores, data.labels, outcome.full_model),
            viz.plot_biplot(outcome.full_scores, data.labels, outcome.full_model, names),
            viz.plot_loadings(outcome.full_model, names),
        ]
        return [p for p in paths if p is not None]  # Hata yönetimi aspect'i başarısız grafik için None döndürür.

    def export_results(
        self, summary: DatasetSummary, data: PreparedData, outcome: PCAOutcome, verification: dict[str, Any]
    ) -> dict[str, Path]:
        """ADIM 7 - Sonuçların dosyaya aktarılması"""
        out = self.config.results_dir
        out.mkdir(parents=True, exist_ok=True)

        k = outcome.model.components_.shape[0]
        scores = pd.DataFrame(outcome.scores, columns=[f"PC{i + 1}" for i in range(k)])
        scores.insert(0, "species", data.labels)

        full = outcome.full_model
        loadings = pd.DataFrame(full.loadings_, index=data.feature_names,
                                columns=[f"PC{i + 1}" for i in range(full.loadings_.shape[1])])
        pca_summary = {
            "n_components": k,
            "ozdegerler": full.eigenvalues_.round(6).tolist(),
            "aciklanan_varyans_orani": (full.eigenvalues_ / full.eigenvalues_.sum()).round(6).tolist(),
            "kumulatif_varyans_orani": np.cumsum(full.eigenvalues_ / full.eigenvalues_.sum()).round(6).tolist(),
            "secilen_modelin_varyans_orani": float(outcome.model.cumulative_variance_ratio_[-1]),
            "yeniden_yapilandirma_mse": outcome.reconstruction_mse,
            "yuzde95_icin_gereken_bilesen": outcome.components_for_95,
            "bilesenler": {f"PC{i + 1}": dict(zip(data.feature_names, row.round(6).tolist()))
                           for i, row in enumerate(full.components_)},
            "kovaryans_matrisi": full.covariance_.round(6).tolist(),
            "sklearn_dogrulama": verification,
            "on_isleme": asdict(data.report),
        }

        files = {
            "dataset_summary": out / "dataset_summary.json",
            "pca_summary": out / "pca_summary.json",
            "pca_scores": out / "pca_scores.csv",
            "loadings": out / "pca_loadings.csv",
        }
        files["dataset_summary"].write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        files["pca_summary"].write_text(json.dumps(pca_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        scores.to_csv(files["pca_scores"], index=False, float_format="%.6f")
        loadings.to_csv(files["loadings"], float_format="%.6f")
        return files

    # ------------------------------------------------------------------ orkestrasyon
    def run(self) -> PipelineResult:
        """Iris PCA iş akışının uçtan uca çalıştırılması"""
        frame = self.load_data()
        summary = self.describe_data(frame)
        data = self.preprocess(frame)
        outcome = self.apply_pca(data.X_scaled)
        verification = self.verify_with_sklearn(data.X_scaled, outcome) if self.config.verify_with_sklearn else {}
        figures = self.visualize(frame, data, outcome) if self.config.make_plots else []
        files = self.export_results(summary, data, outcome, verification)
        return PipelineResult(summary, data, outcome, verification, figures, files)
