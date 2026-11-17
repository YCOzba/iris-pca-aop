"""Veri seti, PCA sonuçları ve performans metriklerinin görselleştirilmesi."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # Ekran gerektirmeyen arka uç: sunucuda/CI'da da çalışır.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401  (3B projeksiyonu kaydeder)

from ..aop.metrics import MetricsRegistry  # noqa: E402
from ..data.loader import FEATURE_COLUMNS, SPECIES, TARGET_COLUMN  # noqa: E402
from ..pca.pca import PCA  # noqa: E402

SPECIES_COLORS = {"setosa": "#1b9e77", "versicolor": "#d95f02", "virginica": "#7570b3"}
SPECIES_MARKERS = {"setosa": "o", "versicolor": "s", "virginica": "^"}
_PRETTY = {
    "sepal_length": "Sepal uzunluk",
    "sepal_width": "Sepal genişlik",
    "petal_length": "Petal uzunluk",
    "petal_width": "Petal genişlik",
}


class IrisVisualizer:
    """Tüm grafikleri ``output_dir`` klasörüne PNG olarak kaydeder."""

    def __init__(self, output_dir: str | Path, dpi: int = 130) -> None:
        self.output_dir = Path(output_dir)
        self.dpi = dpi

    # ------------------------------------------------------------------ yardımcı
    def _save(self, fig: plt.Figure, name: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------ veri seti
    def plot_feature_distributions(self, frame: pd.DataFrame) -> Path:
        """Özniteliklerin türlere göre dağılım grafiğinin çizilmesi"""
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, col in zip(axes.ravel(), FEATURE_COLUMNS):
            bins = np.linspace(frame[col].min(), frame[col].max(), 20)
            for species in SPECIES:
                values = frame.loc[frame[TARGET_COLUMN] == species, col]
                ax.hist(values, bins=bins, alpha=0.55, color=SPECIES_COLORS[species], label=species, edgecolor="white")
            ax.set_title(_PRETTY[col])
            ax.set_xlabel("cm")
            ax.set_ylabel("Frekans")
        axes[0, 0].legend(title="Tür")
        fig.suptitle("Iris öznitelik dağılımları (türlere göre)", fontsize=14, fontweight="bold")
        return self._save(fig, "01_feature_distributions.png")

    def plot_scatter_matrix(self, frame: pd.DataFrame) -> Path:
        """Öznitelik çiftlerinin saçılım matrisinin çizilmesi"""
        n = len(FEATURE_COLUMNS)
        fig, axes = plt.subplots(n, n, figsize=(11, 11))
        for i, row in enumerate(FEATURE_COLUMNS):
            for j, col in enumerate(FEATURE_COLUMNS):
                ax = axes[i, j]
                for species in SPECIES:
                    sub = frame[frame[TARGET_COLUMN] == species]
                    if i == j:
                        ax.hist(sub[col], bins=15, alpha=0.6, color=SPECIES_COLORS[species])
                    else:
                        ax.scatter(sub[col], sub[row], s=10, alpha=0.7, color=SPECIES_COLORS[species], label=species)
                if i == n - 1:
                    ax.set_xlabel(_PRETTY[col])
                if j == 0:
                    ax.set_ylabel(_PRETTY[row])
        handles = [plt.Line2D([], [], marker="o", ls="", color=SPECIES_COLORS[s], label=s) for s in SPECIES]
        fig.legend(handles=handles, loc="upper right", title="Tür")
        fig.suptitle("Iris saçılım matrisi (pair plot)", fontsize=14, fontweight="bold")
        return self._save(fig, "02_scatter_matrix.png")

    def plot_correlation_matrix(self, frame: pd.DataFrame) -> Path:
        """Öznitelikler arası korelasyon matrisinin çizilmesi"""
        corr = frame[FEATURE_COLUMNS].corr().to_numpy()
        labels = [_PRETTY[c] for c in FEATURE_COLUMNS]
        fig, ax = plt.subplots(figsize=(7, 6))
        image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
        ax.set_yticks(range(len(labels)), labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                        color="white" if abs(corr[i, j]) > 0.6 else "black", fontsize=11)
        fig.colorbar(image, ax=ax, label="Pearson korelasyonu")
        ax.set_title("Korelasyon matrisi", fontsize=13, fontweight="bold")
        return self._save(fig, "03_correlation_matrix.png")

    # ------------------------------------------------------------------ PCA
    def plot_scree(self, pca: PCA) -> Path:
        """Açıklanan varyans (scree) grafiğinin çizilmesi"""
        ratios = pca.eigenvalues_ / pca.eigenvalues_.sum()
        cumulative = np.cumsum(ratios)
        idx = np.arange(1, len(ratios) + 1)
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(idx, ratios * 100, color="#4c72b0", label="Bileşen başına")
        ax.plot(idx, cumulative * 100, "o-", color="#c44e52", lw=2, label="Kümülatif")
        ax.axhline(95, ls="--", color="gray", lw=1)
        ax.text(len(idx) + 0.35, 95, "%95", va="center", color="gray")
        for bar, value in zip(bars, ratios):
            inside = value > 0.15
            ax.text(bar.get_x() + bar.get_width() / 2, value * 100 - 4 if inside else value * 100 + 1.5,
                    f"%{value * 100:.1f}", ha="center", color="white" if inside else "black", fontweight="bold")
        ax.set_xticks(idx, [f"PC{i}" for i in idx])
        ax.set_ylim(0, 108)
        ax.set_ylabel("Açıklanan varyans (%)")
        ax.set_title("Scree grafiği — açıklanan varyans oranı", fontsize=13, fontweight="bold")
        ax.legend(loc="center right")
        return self._save(fig, "04_scree_plot.png")

    def plot_pca_2d(self, scores: np.ndarray, labels: Sequence[str], pca: PCA) -> Path:
        """İlk iki temel bileşen üzerinde 2B saçılım grafiğinin çizilmesi"""
        labels = np.asarray(labels)
        ratio = pca.explained_variance_ratio_
        fig, ax = plt.subplots(figsize=(8, 6))
        for species in SPECIES:
            mask = labels == species
            ax.scatter(scores[mask, 0], scores[mask, 1], c=SPECIES_COLORS[species], marker=SPECIES_MARKERS[species],
                       label=species, s=45, alpha=0.8, edgecolor="white", linewidth=0.5)
            center = scores[mask, :2].mean(axis=0)
            ax.scatter(*center, c=SPECIES_COLORS[species], marker="X", s=220, edgecolor="black")
        ax.axhline(0, color="gray", lw=0.6)
        ax.axvline(0, color="gray", lw=0.6)
        ax.set_xlabel(f"PC1 (%{ratio[0] * 100:.1f} varyans)")
        ax.set_ylabel(f"PC2 (%{ratio[1] * 100:.1f} varyans)")
        ax.set_title(f"Iris — PCA 2B izdüşüm (toplam %{ratio[:2].sum() * 100:.1f})", fontsize=13, fontweight="bold")
        ax.legend(title="Tür (X = sınıf merkezi)")
        return self._save(fig, "05_pca_2d.png")

    def plot_pca_3d(self, scores: np.ndarray, labels: Sequence[str], pca: PCA) -> Path:
        """İlk üç temel bileşen üzerinde 3B saçılım grafiğinin çizilmesi"""
        labels = np.asarray(labels)
        ratio = pca.explained_variance_ratio_
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        for species in SPECIES:
            mask = labels == species
            ax.scatter(scores[mask, 0], scores[mask, 1], scores[mask, 2], c=SPECIES_COLORS[species],
                       marker=SPECIES_MARKERS[species], label=species, s=35, alpha=0.85)
        ax.set_xlabel(f"PC1 (%{ratio[0] * 100:.1f})")
        ax.set_ylabel(f"PC2 (%{ratio[1] * 100:.1f})")
        ax.set_zlabel(f"PC3 (%{ratio[2] * 100:.1f})")
        ax.view_init(elev=22, azim=-55)
        ax.set_title(f"Iris — PCA 3B izdüşüm (toplam %{ratio[:3].sum() * 100:.1f})", fontsize=13, fontweight="bold")
        ax.legend()
        return self._save(fig, "06_pca_3d.png")

    def plot_biplot(self, scores: np.ndarray, labels: Sequence[str], pca: PCA, feature_names: Sequence[str]) -> Path:
        """Örnekler ve öznitelik yüklemelerini birlikte gösteren biplot çizimi"""
        labels = np.asarray(labels)
        loadings = pca.loadings_[:, :2]
        scale = np.abs(scores[:, :2]).max() / np.abs(loadings).max() * 0.8
        fig, ax = plt.subplots(figsize=(8, 6.5))
        for species in SPECIES:
            mask = labels == species
            ax.scatter(scores[mask, 0], scores[mask, 1], c=SPECIES_COLORS[species], s=25, alpha=0.45, label=species)
        label_pos = loadings * scale * 1.15
        # Birbirine çok yakın etiketleri dikeyde ayır (ör. petal uzunluk/genişlik).
        min_gap = 0.09 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        order = np.argsort(label_pos[:, 1])
        for prev, cur in zip(order[:-1], order[1:]):
            close_x = abs(label_pos[cur, 0] - label_pos[prev, 0]) < 1.0
            if close_x and label_pos[cur, 1] - label_pos[prev, 1] < min_gap:
                label_pos[cur, 1] = label_pos[prev, 1] + min_gap
        for name, (lx, ly), (tx, ty) in zip(feature_names, loadings, label_pos):
            ax.annotate("", xy=(lx * scale, ly * scale), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.8))
            ax.text(tx, ty, _PRETTY.get(name, name), fontsize=10,
                    fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8))
        ax.axhline(0, color="gray", lw=0.6)
        ax.axvline(0, color="gray", lw=0.6)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Biplot — örnekler ve öznitelik yüklemeleri", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right")
        return self._save(fig, "07_biplot.png")

    def plot_loadings(self, pca: PCA, feature_names: Sequence[str]) -> Path:
        """Bileşen yükleme matrisinin ısı haritasının çizilmesi"""
        loadings = pca.loadings_
        n_comp = loadings.shape[1]
        fig, ax = plt.subplots(figsize=(1.6 * n_comp + 3, 4.5))
        image = ax.imshow(loadings, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n_comp), [f"PC{i + 1}" for i in range(n_comp)])
        ax.set_yticks(range(len(feature_names)), [_PRETTY.get(n, n) for n in feature_names])
        for i in range(loadings.shape[0]):
            for j in range(n_comp):
                ax.text(j, i, f"{loadings[i, j]:.2f}", ha="center", va="center",
                        color="white" if abs(loadings[i, j]) > 0.6 else "black")
        fig.colorbar(image, ax=ax, label="Yükleme (korelasyon)")
        ax.set_title("Yükleme matrisi (loadings)", fontsize=13, fontweight="bold")
        return self._save(fig, "08_loadings_heatmap.png")

    # ------------------------------------------------------------------ AOP metrikleri
    def plot_performance(self, registry: MetricsRegistry) -> Path:
        """AOP ile toplanan performans metriklerinin grafiğinin çizilmesi"""
        summary = registry.summary_frame()
        calls = registry.calls_frame()
        if summary.empty:
            raise ValueError("Çizilecek performans metriği yok")
        summary = summary.sort_values("ortalama_ms")

        fig, (ax_bar, ax_time) = plt.subplots(1, 2, figsize=(15, 0.42 * len(summary) + 2.5),
                                              gridspec_kw={"width_ratios": [1, 1.25]})
        ax_bar.barh(summary["metod"], summary["ortalama_ms"], xerr=summary["std_ms"], color="#4c72b0",
                    ecolor="gray", capsize=3)
        for y, (mean, n) in enumerate(zip(summary["ortalama_ms"], summary["cagri"])):
            ax_bar.text(mean, y, f"  {mean:.2f} ms  (x{n})", va="center", fontsize=8)
        ax_bar.set_xlabel("Ortalama süre (ms) — hata çubuğu: std")
        ax_bar.set_title("Metod başına ortalama süre", fontweight="bold")
        ax_bar.margins(x=0.35)

        order = {m: i for i, m in enumerate(summary["metod"])}
        for _, call in calls.iterrows():
            color = "#55a868" if call["success"] else "#c44e52"
            ax_time.barh(order[call["method"]], call["wall_time"] * 1000, left=call["started_at"] * 1000,
                         color=color, alpha=0.75, edgecolor="black", linewidth=0.3)
        ax_time.set_yticks(range(len(order)), list(order))
        ax_time.set_xlabel("Pipeline başlangıcından itibaren zaman (ms)")
        ax_time.set_title("Zaman çizelgesi (yeşil: başarılı, kırmızı: hata)", fontweight="bold")
        fig.suptitle("AOP ile toplanan performans metrikleri", fontsize=14, fontweight="bold")
        return self._save(fig, "09_performance_metrics.png")
