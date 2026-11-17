"""Bileşim kökü (composition root): aspect'ler burada oluşturulur ve dokunur.

Uygulamanın **tek** AOP yapılandırma noktası burasıdır. Hangi aspect'in
hangi sınıflara, hangi sırayla uygulanacağı buradan okunabilir; iş
sınıflarının kendisi değişmeden kalır.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .aop import MetricsRegistry, weave
from .aop.aspect import Aspect
from .aop.aspects import (
    ErrorHandlingAspect,
    LoggingAspect,
    PerformanceAspect,
    TimingAspect,
    ValidationAspect,
)
from .aop.joinpoint import JoinPoint
from .config import PipelineConfig
from .data.loader import IrisDataLoader
from .data.preprocessing import IrisPreprocessor, StandardScaler
from .logging_config import LOGGER_NAME
from .pca.pca import PCA
from .pipeline import IrisPCAPipeline
from .validation.rules import default_rules
from .visualization.plots import IrisVisualizer


def _skip_plot(jp: JoinPoint, exc: BaseException) -> None:
    """Bir grafik çizilemezse ``None`` döndür; pipeline çalışmaya devam etsin."""
    return None


@dataclass
class Application:
    """Dokunmuş (woven) pipeline ve ona bağlı nesneler."""

    pipeline: IrisPCAPipeline
    registry: MetricsRegistry
    aspects: list[Aspect]
    woven_classes: dict[str, type[Any]]
    logger: logging.Logger


def build_aspects(config: PipelineConfig, registry: MetricsRegistry, logger: logging.Logger) -> list[Aspect]:
    """Beş aspect'i yapılandırarak oluşturur."""
    pre_rules, post_rules = default_rules()
    return [
        LoggingAspect(logger),                                             # order 10
        PerformanceAspect(registry, memory_exclude=config.memory_exclude),  # order 20
        TimingAspect(logger, registry, slow_threshold_ms=config.slow_threshold_ms),  # order 30
        ErrorHandlingAspect(                                               # order 40
            logger,
            registry,
            max_retries=config.max_retries,
            fallbacks={"IrisVisualizer.plot_*": _skip_plot},
        ),
        ValidationAspect(logger, registry, pre=pre_rules, post=post_rules),  # order 50
    ]


def build_application(config: PipelineConfig, registry: MetricsRegistry | None = None) -> Application:
    """Tüm sınıfları aspect'lerle dokur ve kullanıma hazır pipeline'ı döndürür."""
    registry = registry or MetricsRegistry()
    logger = logging.getLogger(f"{LOGGER_NAME}.aop")
    aspects = build_aspects(config, registry, logger)

    woven = {
        cls.__name__: weave(cls, aspects)
        for cls in (IrisDataLoader, IrisPreprocessor, StandardScaler, PCA, IrisVisualizer, IrisPCAPipeline)
    }

    pipeline = woven["IrisPCAPipeline"](
        config=config,
        loader=woven["IrisDataLoader"](config.data_path),
        preprocessor=woven["IrisPreprocessor"](
            drop_duplicates=config.drop_duplicates, scaler=woven["StandardScaler"]()
        ),
        pca_factory=woven["PCA"],
        visualizer=woven["IrisVisualizer"](config.figures_dir),
        registry=registry,
    )
    return Application(pipeline, registry, aspects, woven, logger)
