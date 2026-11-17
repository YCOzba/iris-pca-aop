"""Log yönetimi: çok hedefli, seviyeli, dönen (rotating) ve yapılandırılmış loglar.

Üretilen log hedefleri:

============================  =========  ==============================================
Hedef                         Seviye     Amaç
============================  =========  ==============================================
Konsol (stdout)               INFO*      Kullanıcının canlı takip etmesi
``logs/pipeline.log``         DEBUG      Ayrıntılı, insan tarafından okunur tam kayıt
``logs/errors.log``           WARNING    Yalnızca uyarı ve hatalar (hızlı inceleme)
``logs/pipeline.jsonl``       DEBUG      Satır başına bir JSON; makine ile analiz için
============================  =========  ==============================================

\\* Konsol seviyesi ``--log-level`` ile değiştirilebilir.

Dosyalar ``RotatingFileHandler`` ile 1 MB'ı geçince döndürülür ve en fazla
5 yedek tutulur; böylece loglar diski sınırsız doldurmaz.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOGGER_NAME = "iris_pca_aop"

_STANDARD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Log kaydını, ``extra`` ile gelen alanlar dâhil tek satır JSON'a çevirir."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage().strip(),
        }
        payload.update({k: v for k, v in vars(record).items() if k not in _STANDARD_ATTRS})
        if record.exc_info:
            payload["traceback"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(log_dir: str | Path = "logs", console_level: str = "INFO") -> logging.Logger:
    """Proje logger'ını yapılandırır ve döndürür (tekrar çağrılırsa handler'ları yeniler)."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    detailed = logging.Formatter(
        "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    compact = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # Windows konsolunda Türkçe karakterler için
        except (ValueError, OSError):  # pragma: no cover
            pass
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    console.setFormatter(compact)

    def rotating(name: str, level: int, formatter: logging.Formatter) -> RotatingFileHandler:
        handler = RotatingFileHandler(log_dir / name, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(formatter)
        return handler

    logger.addHandler(console)
    logger.addHandler(rotating("pipeline.log", logging.DEBUG, detailed))
    logger.addHandler(rotating("errors.log", logging.WARNING, detailed))
    logger.addHandler(rotating("pipeline.jsonl", logging.DEBUG, JsonFormatter()))
    return logger


def close_logging() -> None:
    """Tüm handler'ları kapatır (testlerde dosya kilidini bırakmak için)."""
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
