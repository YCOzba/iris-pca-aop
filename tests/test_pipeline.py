"""Uçtan uca pipeline ve doğrulama kuralı testleri."""

import json

import numpy as np
import pytest

from iris_pca_aop.bootstrap import build_application
from iris_pca_aop.config import PipelineConfig
from iris_pca_aop.exceptions import DataLoadError, DataValidationError
from iris_pca_aop.logging_config import close_logging, setup_logging


@pytest.fixture
def app(tmp_path):
    setup_logging(tmp_path / "logs", "WARNING")
    config = PipelineConfig(output_dir=tmp_path / "out", log_dir=tmp_path / "logs", make_plots=False)
    yield build_application(config)
    close_logging()


def test_end_to_end_run(app, tmp_path):
    result = app.pipeline.run()

    assert result.summary.n_samples == 150
    assert result.summary.class_distribution == {"setosa": 50, "versicolor": 50, "virginica": 50}
    assert result.pca.scores.shape == (150, 2)
    assert result.pca.model.cumulative_variance_ratio_[-1] == pytest.approx(0.9581, abs=1e-4)
    assert result.verification["eslesiyor"] is True

    summary = json.loads((tmp_path / "out" / "results" / "pca_summary.json").read_text(encoding="utf-8"))
    assert summary["yuzde95_icin_gereken_bilesen"] == 2

    registry = app.registry
    assert registry.stats["IrisPCAPipeline.run"].calls == 1
    assert registry.stats["PCA.fit"].calls == 2
    assert registry.validations and all(v.passed for v in registry.validations)

    paths = registry.export(tmp_path / "metrics")
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert data["genel"]["hatali_cagri"] == 0

    log_text = (tmp_path / "logs" / "pipeline.log").read_text(encoding="utf-8")
    assert "ADIM 4 - PCA analizinin uygulanması" in log_text
    jsonl = (tmp_path / "logs" / "pipeline.jsonl").read_text(encoding="utf-8").splitlines()
    assert all(json.loads(line)["level"] for line in jsonl)


def test_plots_are_created(tmp_path):
    setup_logging(tmp_path / "logs", "WARNING")
    config = PipelineConfig(output_dir=tmp_path / "out", log_dir=tmp_path / "logs")
    app = build_application(config)
    result = app.pipeline.run()
    close_logging()
    assert len(result.figures) == 8
    assert all(p.exists() and p.stat().st_size > 10_000 for p in result.figures)


def test_validation_rejects_nan(app):
    frame = app.pipeline.load_data()
    frame.loc[0, "petal_length"] = np.nan
    with pytest.raises(DataValidationError, match="Eksik"):
        app.pipeline.preprocess(frame)
    assert any(not v.passed and v.rule == "eksik_deger_yok" for v in app.registry.validations)


def test_validation_rejects_bad_n_components(app):
    with pytest.raises(DataValidationError, match="n_components"):
        app.woven_classes["PCA"](n_components=7).fit(np.random.rand(30, 4))


def test_missing_csv_raises_data_load_error(tmp_path):
    setup_logging(tmp_path / "logs", "WARNING")
    config = PipelineConfig(output_dir=tmp_path / "out", data_path=tmp_path / "yok.csv", make_plots=False)
    app = build_application(config)
    with pytest.raises(DataLoadError):
        app.pipeline.run()
    close_logging()
    assert app.registry.stats["IrisDataLoader.load"].failures == 1


def test_csv_source(tmp_path, app):
    frame = app.pipeline.load_data()
    csv = tmp_path / "iris.csv"
    frame.to_csv(csv, index=False)
    config = PipelineConfig(output_dir=tmp_path / "out2", data_path=csv, make_plots=False)
    result = build_application(config).pipeline.run()
    assert result.summary.n_samples == 150
