"""Pipeline yapılandırması."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelineConfig:
    """Komut satırından veya testlerden ayarlanabilen tüm parametreler."""

    n_components: int = 2
    """Ana PCA modelinde tutulacak bileşen sayısı."""

    output_dir: Path = Path("outputs")
    log_dir: Path = Path("logs")
    log_level: str = "INFO"
    data_path: Path | None = None
    """Verilirse Iris verisi bu CSV'den okunur; yoksa scikit-learn'deki kopya kullanılır."""

    drop_duplicates: bool = False
    make_plots: bool = True
    slow_threshold_ms: float = 1000.0
    """Bu süreyi aşan çağrılar ``WARNING`` olarak loglanır."""

    max_retries: int = 2
    """Geçici (I/O) hatalarda yeniden deneme sayısı."""

    verify_with_sklearn: bool = True

    memory_exclude: tuple[str, ...] = ("IrisVisualizer.*", "IrisPCAPipeline.visualize", "IrisPCAPipeline.run")
    """Bellek ölçümü (tracemalloc) yapılmayacak metod desenleri; tracemalloc
    matplotlib çizimlerini belirgin biçimde yavaşlattığı için grafikler hariç tutulur."""

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self.log_dir = Path(self.log_dir)
        if self.data_path is not None:
            self.data_path = Path(self.data_path)

    @property
    def figures_dir(self) -> Path:
        return self.output_dir / "figures"

    @property
    def results_dir(self) -> Path:
        return self.output_dir / "results"

    @property
    def metrics_dir(self) -> Path:
        return self.output_dir / "metrics"
