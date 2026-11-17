"""Iris veri setinin yüklenmesi ve tanıtılması.

Bu modülde **hiçbir loglama, zamanlama veya doğrulama kodu yoktur**;
bunların tamamı aspect'ler tarafından dışarıdan eklenir.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.datasets import load_iris

from ..exceptions import DataLoadError

FEATURE_COLUMNS: list[str] = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET_COLUMN = "species"
SPECIES: list[str] = ["setosa", "versicolor", "virginica"]

FEATURE_DESCRIPTIONS: dict[str, str] = {
    "sepal_length": "Çanak yaprak (sepal) uzunluğu (cm)",
    "sepal_width": "Çanak yaprak (sepal) genişliği (cm)",
    "petal_length": "Taç yaprak (petal) uzunluğu (cm)",
    "petal_width": "Taç yaprak (petal) genişliği (cm)",
}

# scikit-learn'deki sütun adlarını projedeki sade adlara eşler.
_SKLEARN_COLUMN_MAP = {
    "sepal length (cm)": "sepal_length",
    "sepal width (cm)": "sepal_width",
    "petal length (cm)": "petal_length",
    "petal width (cm)": "petal_width",
}


@dataclass
class DatasetSummary:
    """Veri setinin hangi verileri kapsadığını özetler."""

    n_samples: int
    n_features: int
    feature_names: list[str]
    feature_descriptions: dict[str, str]
    classes: list[str]
    class_distribution: dict[str, int]
    missing_values: dict[str, int]
    duplicate_rows: int
    statistics: dict[str, dict[str, float]]
    class_means: dict[str, dict[str, float]]
    correlation: dict[str, dict[str, float]]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IrisDataLoader:
    """Iris veri setini scikit-learn'den veya bir CSV dosyasından yükler.

    Parameters
    ----------
    csv_path:
        Verilirse veri bu CSV dosyasından okunur. CSV, ``FEATURE_COLUMNS`` ve
        ``species`` sütunlarını içermelidir. Verilmezse scikit-learn ile gelen
        (internet gerektirmeyen) gömülü kopya kullanılır.
    """

    def __init__(self, csv_path: str | Path | None = None) -> None:
        self.csv_path = Path(csv_path) if csv_path else None

    def load(self) -> pd.DataFrame:
        """Iris veri setinin yüklenmesi"""
        if self.csv_path is not None:
            return self._load_csv(self.csv_path)
        return self._load_sklearn()

    @staticmethod
    def _load_sklearn() -> pd.DataFrame:
        bunch = load_iris(as_frame=True)
        frame = bunch.frame.rename(columns=_SKLEARN_COLUMN_MAP)
        frame[TARGET_COLUMN] = pd.Categorical.from_codes(frame.pop("target"), categories=SPECIES)
        return frame[FEATURE_COLUMNS + [TARGET_COLUMN]]

    @staticmethod
    def _load_csv(path: Path) -> pd.DataFrame:
        if not path.exists():
            raise DataLoadError("Veri dosyası bulunamadı", context={"yol": str(path)})
        frame = pd.read_csv(path)
        missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in frame.columns]
        if missing:
            raise DataLoadError("CSV dosyasında gerekli sütunlar eksik", context={"eksik": missing})
        frame[TARGET_COLUMN] = frame[TARGET_COLUMN].astype(str).str.replace("Iris-", "", regex=False)
        frame[TARGET_COLUMN] = pd.Categorical(frame[TARGET_COLUMN], categories=SPECIES)
        return frame[FEATURE_COLUMNS + [TARGET_COLUMN]]

    def describe(self, frame: pd.DataFrame) -> DatasetSummary:
        """Veri setinin hangi verileri kapsadığının çıkarılması"""
        features = frame[FEATURE_COLUMNS]
        stats = features.describe().T
        stats["variance"] = features.var()
        stats["skewness"] = features.skew()
        distribution = frame[TARGET_COLUMN].value_counts().reindex(SPECIES, fill_value=0)

        duplicates = int(frame.duplicated().sum())
        notes = [
            "150 örnek, 4 sayısal öznitelik ve 3 sınıftan oluşan dengeli bir veri seti.",
            "Taç yaprak (petal) ölçümleri türleri ayırt etmede en bilgilendirici özniteliklerdir.",
        ]
        if duplicates:
            notes.append(f"{duplicates} adet tekrar eden satır var; gerçek gözlem olabileceği için silinmedi.")

        return DatasetSummary(
            n_samples=len(frame),
            n_features=len(FEATURE_COLUMNS),
            feature_names=list(FEATURE_COLUMNS),
            feature_descriptions=dict(FEATURE_DESCRIPTIONS),
            classes=list(SPECIES),
            class_distribution={k: int(v) for k, v in distribution.items()},
            missing_values={k: int(v) for k, v in frame.isna().sum().items()},
            duplicate_rows=duplicates,
            statistics={
                col: {k: round(float(v), 4) for k, v in row.items()} for col, row in stats.iterrows()
            },
            class_means={
                str(species): {k: round(float(v), 4) for k, v in row.items()}
                for species, row in frame.groupby(TARGET_COLUMN, observed=True)[FEATURE_COLUMNS].mean().iterrows()
            },
            correlation={
                col: {k: round(float(v), 4) for k, v in row.items()} for col, row in features.corr().iterrows()
            },
            notes=notes,
        )
