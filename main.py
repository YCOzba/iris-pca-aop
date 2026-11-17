"""Komut satırı giriş noktası.

Örnekler::

    python main.py                       # tam pipeline (2 bileşen)
    python main.py --n-components 3      # 3 bileşenli PCA
    python main.py --log-level DEBUG     # zamanlama/doğrulama ayrıntıları da konsolda
    python main.py --demo-errors         # hata yakalama senaryolarını da çalıştır
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from iris_pca_aop.bootstrap import build_application
from iris_pca_aop.config import PipelineConfig
from iris_pca_aop.exceptions import IrisPCAError
from iris_pca_aop.logging_config import close_logging, setup_logging


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iris veri seti üzerinde AOP ile izlenen PCA analizi")
    parser.add_argument("--n-components", type=int, default=2, help="PCA bileşen sayısı (varsayılan: 2)")
    parser.add_argument("--data", type=Path, default=None, help="İsteğe bağlı Iris CSV dosyası")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Çıktı klasörü")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"), help="Log klasörü")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Konsol log seviyesi (dosyalara her zaman DEBUG yazılır)")
    parser.add_argument("--slow-ms", type=float, default=1000.0, help="Yavaş çağrı uyarı eşiği (ms)")
    parser.add_argument("--no-plots", action="store_true", help="Grafik üretme")
    parser.add_argument("--demo-errors", action="store_true", help="Hata yakalama senaryolarını çalıştır")
    return parser.parse_args(argv)


def print_report(result, registry) -> None:
    summary, pca = result.summary, result.pca
    ratios = pca.full_model.eigenvalues_ / pca.full_model.eigenvalues_.sum()
    lines = [
        "",
        "=" * 78,
        " IRIS VERI SETI",
        "=" * 78,
        f" Örnek sayısı      : {summary.n_samples}",
        f" Öznitelikler      : {', '.join(summary.feature_names)}",
        f" Sınıf dağılımı    : {summary.class_distribution}",
        f" Eksik değer       : {sum(summary.missing_values.values())}",
        f" Tekrar eden satır : {summary.duplicate_rows}",
        "",
        "=" * 78,
        " PCA SONUÇLARI",
        "=" * 78,
    ]
    for i, (value, ratio) in enumerate(zip(pca.full_model.eigenvalues_, ratios), start=1):
        lines.append(f" PC{i}: özdeğer = {value:7.4f}   açıklanan varyans = %{ratio * 100:6.2f}"
                     f"   kümülatif = %{ratios[:i].sum() * 100:6.2f}")
    lines += [
        f" Seçilen {pca.model.components_.shape[0]} bileşen toplam varyansın %"
        f"{pca.model.cumulative_variance_ratio_[-1] * 100:.2f}'ini açıklıyor.",
        f" %95 varyans için gereken bileşen sayısı: {pca.components_for_95}",
        f" Yeniden yapılandırma hatası (MSE): {pca.reconstruction_mse:.5f}",
    ]
    if result.verification:
        v = result.verification
        lines.append(f" scikit-learn ile eşleşme: {'EVET' if v['eslesiyor'] else 'HAYIR'} "
                     f"(maks. skor farkı = {v['maks_skor_farki']:.2e})")
    lines += ["", "=" * 78, " AOP PERFORMANS METRİKLERİ", "=" * 78, registry.format_table()]
    data = registry.to_dict()["genel"]
    lines += [
        "",
        f" Toplam çağrı: {data['toplam_cagri']} | hatalı: {data['hatali_cagri']} | "
        f"yakalanan hata: {data['yakalanan_hata']} | doğrulama kontrolü: {data['dogrulama_kontrolu']} "
        f"(başarısız: {data['basarisiz_dogrulama']})",
        "=" * 78,
    ]
    print("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = setup_logging(args.log_dir, args.log_level)
    config = PipelineConfig(
        n_components=args.n_components,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        log_level=args.log_level,
        data_path=args.data,
        make_plots=not args.no_plots,
        slow_threshold_ms=args.slow_ms,
    )
    app = build_application(config)
    logger.info("Iris PCA + AOP pipeline başlatılıyor (n_components=%d)", config.n_components)

    exit_code = 0
    try:
        result = app.pipeline.run()
        if config.make_plots:
            perf_plot = app.pipeline.visualizer.plot_performance(app.registry)
            if perf_plot:
                result.figures.append(perf_plot)
        print_report(result, app.registry)
        logger.info("%d grafik kaydedildi: %s", len(result.figures), config.figures_dir)
    except IrisPCAError as exc:
        logger.critical("Pipeline durduruldu: %s", exc)
        exit_code = 1

    if args.demo_errors:
        from iris_pca_aop.demo import run_error_scenarios

        scenarios = run_error_scenarios(app)
        path = config.metrics_dir / "error_scenarios.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(scenarios, ensure_ascii=False, indent=2), encoding="utf-8")

    paths = app.registry.export(config.metrics_dir)
    logger.info("Performans metrikleri kaydedildi: %s", ", ".join(str(p) for p in paths.values()))
    logging.getLogger("iris_pca_aop").info("Loglar: %s", config.log_dir.resolve())
    close_logging()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
