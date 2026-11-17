"""Iris/PCA alanına özgü doğrulama kuralları.

Her kural bir ``JoinPoint`` alır ve ihlal durumunda ``DataValidationError``
fırlatır. Kurallar ``ValidationAspect``'e metod desenleriyle bağlanır
(bkz. :func:`default_rules`). Böylece iş kodunda tek bir ``if`` kontrolü
bile bulunmaz.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..aop.aspects import Rule
from ..aop.joinpoint import JoinPoint
from ..data.loader import FEATURE_COLUMNS, SPECIES, TARGET_COLUMN
from ..exceptions import DataValidationError

# --------------------------------------------------------------------------- yardımcılar


def _frame_arg(jp: JoinPoint) -> pd.DataFrame:
    frame = jp.get_argument("frame", 0)
    if not isinstance(frame, pd.DataFrame):
        raise DataValidationError("Girdi bir pandas DataFrame olmalı", context={"tip": type(frame).__name__})
    return frame


def _matrix_arg(jp: JoinPoint, name: str = "X") -> np.ndarray:
    value = jp.get_argument(name, 0)
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("Girdi sayısal bir matrise dönüştürülemedi", context={"tip": type(value).__name__}) from exc
    if matrix.ndim != 2:
        raise DataValidationError("Girdi 2 boyutlu olmalı (örnek x öznitelik)", context={"boyut": matrix.shape})
    return matrix


# --------------------------------------------------------------------------- DataFrame kuralları


def _not_empty(jp: JoinPoint) -> None:
    frame = jp.result if jp.result is not None else _frame_arg(jp)
    if frame.empty:
        raise DataValidationError("Veri seti boş")


def _schema(jp: JoinPoint) -> None:
    frame = jp.result if jp.result is not None else _frame_arg(jp)
    missing = [c for c in FEATURE_COLUMNS + [TARGET_COLUMN] if c not in frame.columns]
    if missing:
        raise DataValidationError("Gerekli sütunlar eksik", context={"eksik": missing})
    non_numeric = [c for c in FEATURE_COLUMNS if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise DataValidationError("Öznitelik sütunları sayısal olmalı", context={"sayisal_olmayan": non_numeric})


def _no_missing(jp: JoinPoint) -> None:
    frame = _frame_arg(jp)
    counts = frame[FEATURE_COLUMNS + [TARGET_COLUMN]].isna().sum()
    counts = counts[counts > 0]
    if not counts.empty:
        raise DataValidationError("Eksik (NaN) değer bulundu", context={"eksik_degerler": counts.to_dict()})


def _finite_and_positive(jp: JoinPoint) -> None:
    values = _frame_arg(jp)[FEATURE_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise DataValidationError("Sonsuz (inf) değer bulundu")
    if (values <= 0).any():
        raise DataValidationError(
            "Çiçek ölçümleri pozitif olmalı (cm)", context={"hatali_hucre": int((values <= 0).sum())}
        )


def _known_species(jp: JoinPoint) -> None:
    frame = jp.result if jp.result is not None else _frame_arg(jp)
    labels = set(frame[TARGET_COLUMN].dropna().astype(str).unique())
    unknown = labels - set(SPECIES)
    if unknown:
        raise DataValidationError("Bilinmeyen tür etiketi", context={"bilinmeyen": sorted(unknown)})


def _enough_samples(jp: JoinPoint) -> None:
    frame = jp.result
    if len(frame) < 10:
        raise DataValidationError("PCA için en az 10 örnek gerekli", context={"ornek": len(frame)})


# --------------------------------------------------------------------------- matris / PCA kuralları


def _finite_matrix(jp: JoinPoint) -> None:
    matrix = _matrix_arg(jp)
    if matrix.size == 0:
        raise DataValidationError("Matris boş")
    bad = int((~np.isfinite(matrix)).sum())
    if bad:
        raise DataValidationError("Matriste NaN/inf değer var", context={"gecersiz_hucre": bad})


def _n_components_in_range(jp: JoinPoint) -> None:
    pca = jp.target
    n_samples, n_features = _matrix_arg(jp).shape
    if n_samples < 2:
        raise DataValidationError("Kovaryans için en az 2 örnek gerekli", context={"ornek": n_samples})
    k = pca.n_components
    if k is None:
        return
    limit = min(n_samples, n_features)
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= limit:
        raise DataValidationError(
            "n_components 1 ile min(örnek, öznitelik) arasında bir tam sayı olmalı",
            context={"n_components": k, "izin_verilen_aralik": f"1..{limit}"},
        )


def _feature_count_matches(jp: JoinPoint) -> None:
    pca = jp.target
    if pca.n_features_in_ is None:
        return  # Model eğitilmemiş: PCA kendi PCAError'ını fırlatacak.
    n_features = _matrix_arg(jp).shape[1]
    if n_features != pca.n_features_in_:
        raise DataValidationError(
            "Öznitelik sayısı eğitimdekiyle aynı olmalı",
            context={"beklenen": pca.n_features_in_, "gelen": n_features},
        )


def _standardized(jp: JoinPoint) -> None:
    X = jp.result.X_scaled
    if not np.allclose(X.mean(axis=0), 0.0, atol=1e-8) or not np.allclose(X.std(axis=0), 1.0, atol=1e-8):
        raise DataValidationError("Standardizasyon sonrası ortalama 0, std 1 olmalı")


def _variance_ratio_valid(jp: JoinPoint) -> None:
    ratios = jp.target.explained_variance_ratio_
    if (ratios < -1e-12).any() or ratios.sum() > 1 + 1e-9:
        raise DataValidationError("Açıklanan varyans oranları [0, 1] aralığında olmalı", context={"toplam": float(ratios.sum())})
    if (np.diff(ratios) > 1e-12).any():
        raise DataValidationError("Bileşenler varyansa göre azalan sırada olmalı")


def _components_orthonormal(jp: JoinPoint) -> None:
    W = jp.target.components_
    gram = W @ W.T
    if not np.allclose(gram, np.eye(len(W)), atol=1e-8):
        raise DataValidationError("Temel bileşenler ortonormal değil", context={"maks_sapma": float(np.abs(gram - np.eye(len(W))).max())})


def _projection_shape(jp: JoinPoint) -> None:
    Z = jp.result
    n_samples = _matrix_arg(jp).shape[0]
    k = jp.target.components_.shape[0]
    if Z.shape != (n_samples, k) or not np.isfinite(Z).all():
        raise DataValidationError("İzdüşüm boyutu/değerleri hatalı", context={"beklenen": (n_samples, k), "gelen": Z.shape})


# --------------------------------------------------------------------------- kural haritası


def default_rules() -> tuple[dict[str, list[Rule]], dict[str, list[Rule]]]:
    """``ValidationAspect`` için (pre, post) kural sözlüklerini döndürür."""
    pre = {
        "IrisPreprocessor.prepare": [
            Rule("dataframe_bos_degil", _not_empty),
            Rule("sema_ve_sayisal_tipler", _schema),
            Rule("eksik_deger_yok", _no_missing),
            Rule("sonlu_ve_pozitif_olcum", _finite_and_positive),
            Rule("bilinen_tur_etiketleri", _known_species),
        ],
        "PCA.fit": [
            Rule("sonlu_sayisal_matris", _finite_matrix),
            Rule("n_components_araligi", _n_components_in_range),
        ],
        "PCA.transform": [
            Rule("sonlu_sayisal_matris", _finite_matrix),
            Rule("oznitelik_sayisi_uyumu", _feature_count_matches),
        ],
    }
    post = {
        "IrisDataLoader.load": [
            Rule("dataframe_bos_degil", _not_empty),
            Rule("sema_ve_sayisal_tipler", _schema),
            Rule("bilinen_tur_etiketleri", _known_species),
            Rule("yeterli_ornek_sayisi", _enough_samples),
        ],
        "IrisPreprocessor.prepare": [Rule("standardizasyon_kontrolu", _standardized)],
        "PCA.fit": [
            Rule("varyans_orani_gecerli", _variance_ratio_valid),
            Rule("bilesenler_ortonormal", _components_orthonormal),
        ],
        "PCA.transform": [Rule("izdusum_boyutu", _projection_shape)],
    }
    return pre, post
