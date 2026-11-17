"""Performans metriklerinin toplandığı merkezi kayıt (registry).

``TimingAspect``, ``PerformanceAspect`` ve ``ErrorHandlingAspect`` ölçtükleri
değerleri buraya yazar. Çalışmanın sonunda kayıt; tablo, JSON ve CSV olarak
dışa aktarılır ve performans grafiği bu verilerden çizilir.
"""

from __future__ import annotations

import json
import math
import statistics
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class CallRecord:
    """Tek bir metod çağrısının ölçümü."""

    call_id: int
    method: str
    depth: int
    started_at: float
    """Pipeline başlangıcına göre saniye cinsinden başlangıç anı."""
    wall_time: float
    cpu_time: float
    success: bool
    peak_memory_kb: float = 0.0
    memory_delta_kb: float = 0.0
    error: str | None = None


@dataclass
class ErrorRecord:
    """Hata yönetimi aspect'inin yakaladığı bir hata."""

    timestamp: str
    method: str
    error_type: str
    message: str
    handled: bool
    action: str
    attempt: int


@dataclass
class ValidationRecord:
    """Doğrulama aspect'inin çalıştırdığı tek bir kural."""

    method: str
    rule: str
    phase: str
    """``pre`` (girdi) veya ``post`` (çıktı)."""
    passed: bool
    message: str = ""


@dataclass
class MethodStats:
    """Bir metodun tüm çağrılarına ait özet istatistikler."""

    method: str
    calls: int = 0
    successes: int = 0
    failures: int = 0
    total_time: float = 0.0
    min_time: float = math.inf
    max_time: float = 0.0
    times: list[float] = field(default_factory=list, repr=False)
    total_cpu_time: float = 0.0
    peak_memory_kb: float = 0.0
    retries: int = 0

    @property
    def mean_time(self) -> float:
        return self.total_time / self.calls if self.calls else 0.0

    @property
    def std_time(self) -> float:
        return statistics.stdev(self.times) if len(self.times) > 1 else 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.calls if self.calls else 0.0


class MetricsRegistry:
    """Thread-güvenli metrik deposu."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._origin: float | None = None
        self.calls: list[CallRecord] = []
        self.errors: list[ErrorRecord] = []
        self.validations: list[ValidationRecord] = []
        self.stats: dict[str, MethodStats] = {}
        self.slow_calls: list[str] = []

    # ---------------------------------------------------------------- kayıt
    def relative_time(self, absolute: float) -> float:
        """İlk ölçümden itibaren geçen süreyi döndürür (zaman çizelgesi için)."""
        with self._lock:
            if self._origin is None:
                self._origin = absolute
            return absolute - self._origin

    def record_call(self, record: CallRecord) -> None:
        with self._lock:
            self.calls.append(record)
            stats = self.stats.setdefault(record.method, MethodStats(record.method))
            stats.calls += 1
            stats.successes += int(record.success)
            stats.failures += int(not record.success)
            stats.total_time += record.wall_time
            stats.total_cpu_time += record.cpu_time
            stats.min_time = min(stats.min_time, record.wall_time)
            stats.max_time = max(stats.max_time, record.wall_time)
            stats.times.append(record.wall_time)
            stats.peak_memory_kb = max(stats.peak_memory_kb, record.peak_memory_kb)

    def record_error(self, record: ErrorRecord) -> None:
        with self._lock:
            self.errors.append(record)
            if record.action == "retry":
                self.stats.setdefault(record.method, MethodStats(record.method)).retries += 1

    def record_slow_call(self, message: str) -> None:
        with self._lock:
            self.slow_calls.append(message)

    def record_validation(self, record: ValidationRecord) -> None:
        with self._lock:
            self.validations.append(record)

    def reset(self) -> None:
        with self._lock:
            self._origin = None
            self.calls.clear()
            self.errors.clear()
            self.validations.clear()
            self.stats.clear()
            self.slow_calls.clear()

    # ---------------------------------------------------------------- rapor
    def summary_frame(self) -> pd.DataFrame:
        """Metod bazında özet tabloyu (toplam süreye göre azalan) döndürür."""
        rows = [
            {
                "metod": s.method,
                "cagri": s.calls,
                "basarili": s.successes,
                "hatali": s.failures,
                "tekrar_deneme": s.retries,
                "toplam_ms": s.total_time * 1000,
                "ortalama_ms": s.mean_time * 1000,
                "min_ms": (s.min_time if s.calls else 0.0) * 1000,
                "max_ms": s.max_time * 1000,
                "std_ms": s.std_time * 1000,
                "cpu_ms": s.total_cpu_time * 1000,
                "tepe_bellek_kb": s.peak_memory_kb,
                "basari_orani": s.success_rate,
            }
            for s in self.stats.values()
        ]
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame.sort_values("toplam_ms", ascending=False).reset_index(drop=True)
        return frame

    def calls_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(c) for c in self.calls])

    def to_dict(self) -> dict[str, Any]:
        summary = self.summary_frame()
        total_calls = len(self.calls)
        failed = sum(1 for c in self.calls if not c.success)
        return {
            "olusturulma_zamani": datetime.now().isoformat(timespec="seconds"),
            "genel": {
                "toplam_cagri": total_calls,
                "basarili_cagri": total_calls - failed,
                "hatali_cagri": failed,
                "yakalanan_hata": len(self.errors),
                "yavas_cagri": len(self.slow_calls),
                "dogrulama_kontrolu": len(self.validations),
                "basarisiz_dogrulama": sum(1 for v in self.validations if not v.passed),
                "en_dis_cagrilarin_toplam_suresi_ms": sum(c.wall_time for c in self.calls if c.depth == 0) * 1000,
            },
            "metod_ozeti": summary.to_dict(orient="records"),
            "cagrilar": [asdict(c) for c in self.calls],
            "hatalar": [asdict(e) for e in self.errors],
            "dogrulamalar": [asdict(v) for v in self.validations],
            "yavas_cagrilar": list(self.slow_calls),
        }

    def export(self, directory: str | Path) -> dict[str, Path]:
        """Metrikleri JSON + CSV olarak diske yazar ve dosya yollarını döndürür."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "json": directory / "performance_metrics.json",
            "summary_csv": directory / "method_summary.csv",
            "calls_csv": directory / "call_log.csv",
        }
        paths["json"].write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
        self.summary_frame().to_csv(paths["summary_csv"], index=False, float_format="%.4f")
        self.calls_frame().to_csv(paths["calls_csv"], index=False, float_format="%.6f")
        if self.errors:
            paths["errors_csv"] = directory / "error_log.csv"
            pd.DataFrame([asdict(e) for e in self.errors]).to_csv(paths["errors_csv"], index=False)
        if self.validations:
            paths["validations_csv"] = directory / "validation_log.csv"
            pd.DataFrame([asdict(v) for v in self.validations]).to_csv(paths["validations_csv"], index=False)
        return paths

    def format_table(self) -> str:
        """Konsola yazdırılabilir, hizalı özet tablo."""
        frame = self.summary_frame()
        if frame.empty:
            return "(henüz metrik yok)"
        view = frame[["metod", "cagri", "hatali", "tekrar_deneme", "toplam_ms", "ortalama_ms", "max_ms", "tepe_bellek_kb"]]
        return view.to_string(index=False, float_format=lambda v: f"{v:,.2f}")
