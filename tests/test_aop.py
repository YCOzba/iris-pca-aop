"""AOP çatısı ve aspect'lerin davranış testleri."""

import logging

import numpy as np
import pytest

from iris_pca_aop.aop import Aspect, MetricsRegistry, advise, is_woven, weave
from iris_pca_aop.aop.aspects import (
    ErrorHandlingAspect,
    LoggingAspect,
    PerformanceAspect,
    Rule,
    TimingAspect,
    ValidationAspect,
)
from iris_pca_aop.exceptions import DataValidationError, PCAError, PipelineError

LOGGER = logging.getLogger("iris_pca_aop.test")


class Recorder(Aspect):
    def __init__(self, name, events, **kw):
        super().__init__(**kw)
        self.name, self.events = name, events

    def before(self, jp):
        self.events.append(f"{self.name}.before")

    def after_returning(self, jp):
        self.events.append(f"{self.name}.after_returning")

    def after_throwing(self, jp):
        self.events.append(f"{self.name}.after_throwing")

    def after(self, jp):
        self.events.append(f"{self.name}.after")


class Calculator:
    def add(self, a, b):
        """İki sayının toplanması"""
        return a + b

    def divide(self, a, b):
        return a / b

    def _private(self):
        return 1


def test_advice_order_and_nesting():
    events = []
    Woven = weave(Calculator, [Recorder("inner", events, order=2), Recorder("outer", events, order=1)])
    assert Woven().add(1, 2) == 3
    assert events == [
        "outer.before", "inner.before",
        "inner.after_returning", "inner.after",
        "outer.after_returning", "outer.after",
    ]


def test_after_throwing_is_called_and_error_propagates():
    events = []
    Woven = weave(Calculator, [Recorder("r", events)])
    with pytest.raises(ZeroDivisionError):
        Woven().divide(1, 0)
    assert events == ["r.before", "r.after_throwing", "r.after"]


def test_pointcut_selects_methods_and_original_class_untouched():
    Woven = weave(Calculator, [Recorder("r", [], pointcut="Calculator.add")])
    assert is_woven(Woven.add)
    assert not is_woven(Woven.divide)
    assert not is_woven(Woven._private)
    assert not is_woven(Calculator.add)          # orijinal sınıf değişmedi
    assert Woven.__name__ == "Calculator" and issubclass(Woven, Calculator)


def test_advise_decorator_on_function():
    events = []

    @advise(Recorder("r", events))
    def square(x):
        return x * x

    assert square(4) == 16
    assert events[0] == "r.before"


def test_timing_and_performance_collect_metrics():
    registry = MetricsRegistry()
    Woven = weave(Calculator, [PerformanceAspect(registry), TimingAspect(LOGGER, registry, slow_threshold_ms=1e9)])
    calc = Woven()
    for i in range(5):
        calc.add(i, i)
    with pytest.raises(ZeroDivisionError):
        calc.divide(1, 0)

    stats = registry.stats
    assert stats["Calculator.add"].calls == 5
    assert stats["Calculator.add"].successes == 5
    assert stats["Calculator.divide"].failures == 1
    assert stats["Calculator.add"].min_time <= stats["Calculator.add"].mean_time <= stats["Calculator.add"].max_time
    frame = registry.summary_frame()
    assert {"metod", "ortalama_ms", "tepe_bellek_kb"} <= set(frame.columns)


def test_slow_call_is_reported():
    registry = MetricsRegistry()
    Woven = weave(Calculator, [TimingAspect(LOGGER, registry, slow_threshold_ms=0.0)])
    Woven().add(1, 1)
    assert registry.slow_calls and "Calculator.add" in registry.slow_calls[0]


def test_performance_measures_memory():
    registry = MetricsRegistry()

    class Allocator:
        def allocate(self):
            return np.ones(500_000)  # ~3.8 MB

    weave(Allocator, [PerformanceAspect(registry)])().allocate()
    assert registry.stats["Allocator.allocate"].peak_memory_kb > 3000


def test_validation_blocks_invalid_input_and_records():
    registry = MetricsRegistry()

    def positive(jp):
        if jp.args[1] == 0:
            raise DataValidationError("bölen sıfır olamaz")

    Woven = weave(Calculator, [ValidationAspect(LOGGER, registry, pre={"divide": [Rule("bolen_sifir_degil", positive)]})])
    assert Woven().divide(4, 2) == 2
    with pytest.raises(DataValidationError) as info:
        Woven().divide(1, 0)
    assert info.value.context["kural"] == "bolen_sifir_degil"
    assert [v.passed for v in registry.validations] == [True, False]


def test_error_handling_retries_transient_errors():
    registry = MetricsRegistry()
    attempts = {"n": 0}

    class Flaky:
        def read(self):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise OSError("geçici")
            return "veri"

    Woven = weave(Flaky, [ErrorHandlingAspect(LOGGER, registry, max_retries=2, backoff_seconds=0)])
    assert Woven().read() == "veri"
    assert attempts["n"] == 3
    assert [e.action for e in registry.errors] == ["retry", "retry"]


def test_error_handling_gives_up_after_max_retries():
    registry = MetricsRegistry()

    class AlwaysFails:
        def read(self):
            raise OSError("kalıcı")

    with pytest.raises(PipelineError) as info:
        weave(AlwaysFails, [ErrorHandlingAspect(LOGGER, registry, max_retries=1, backoff_seconds=0)])().read()
    assert isinstance(info.value.__cause__, OSError)


def test_error_handling_wraps_unexpected_and_keeps_domain_errors():
    registry = MetricsRegistry()

    class Service:
        def unexpected(self):
            raise KeyError("x")

        def domain(self):
            raise PCAError("alan hatası")

    Woven = weave(Service, [ErrorHandlingAspect(LOGGER, registry)])
    with pytest.raises(PipelineError) as info:
        Woven().unexpected()
    assert isinstance(info.value.__cause__, KeyError)
    with pytest.raises(PCAError):
        Woven().domain()
    assert [e.action for e in registry.errors] == ["wrap", "raise"]


def test_error_handling_fallback():
    registry = MetricsRegistry()
    Woven = weave(Calculator, [ErrorHandlingAspect(LOGGER, registry, fallbacks={"divide": lambda jp, exc: float("inf")})])
    assert Woven().divide(1, 0) == float("inf")
    assert registry.errors[0].action == "fallback"


def test_nested_error_is_recorded_once():
    registry = MetricsRegistry()

    class Outer:
        def outer(self):
            return self.inner()

        def inner(self):
            raise ValueError("iç hata")

    with pytest.raises(PipelineError):
        weave(Outer, [ErrorHandlingAspect(LOGGER, registry)])().outer()
    assert len(registry.errors) == 1 and registry.errors[0].method == "Outer.inner"


def test_logging_aspect_writes_start_end_and_error(caplog):
    Woven = weave(Calculator, [LoggingAspect(LOGGER)])
    with caplog.at_level(logging.INFO, logger=LOGGER.name):
        Woven().add(1, 2)
        with pytest.raises(ZeroDivisionError):
            Woven().divide(1, 0)
    text = caplog.text
    assert "Calculator.add başladı: İki sayının toplanması" in text
    assert "Calculator.add tamamlandı" in text
    assert "Calculator.divide BAŞARISIZ: ZeroDivisionError" in text
    events = [getattr(r, "event", None) for r in caplog.records]
    assert events == ["start", "end", "start", "error"]
