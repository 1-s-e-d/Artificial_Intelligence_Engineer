"""HTTP API для оценки качества датасетов."""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .core import compute_quality_flags, get_basic_stats, get_missing_info

# ---------- Настройка структурированного логирования (Вариант D) ----------

# Создаем папку для логов
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# Настраиваем логгер
logger = logging.getLogger("eda_api")
logger.setLevel(logging.INFO)

# Хэндлер для файла (JSON формат)
file_handler = logging.FileHandler(LOGS_DIR / "api.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)

# Хэндлер для консоли (обычный формат)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Форматтеры
file_formatter = logging.Formatter('%(message)s')
console_formatter = logging.Formatter('[%(levelname)s] %(message)s')

file_handler.setFormatter(file_formatter)
console_handler.setFormatter(console_formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


def log_request(
    endpoint: str,
    status: int,
    latency_ms: float,
    request_id: str,
    **kwargs
) -> None:
    """
    Логирует запрос в структурированном JSON формате.

    Параметры:
        endpoint: название эндпоинта
        status: HTTP статус ответа
        latency_ms: время обработки в миллисекундах
        request_id: уникальный ID запроса
        **kwargs: дополнительные поля (n_rows, n_cols, ok_for_model и т.д.)
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "request_id": request_id,
        "endpoint": endpoint,
        "status": status,
        "latency_ms": round(latency_ms, 2),
        **kwargs
    }

    # Пишем в файл как JSON
    logger.info(json.dumps(log_entry, ensure_ascii=False))


# ---------- Вспомогательная функция для конвертации типов ----------

def convert_to_native_types(obj: Any) -> Any:
    """
    Рекурсивно конвертирует numpy/pandas типы в нативные Python типы.
    Необходимо для корректной сериализации в JSON через Pydantic.
    """
    import numpy as np

    if isinstance(obj, dict):
        return {key: convert_to_native_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_, np.bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


app = FastAPI(
    title="AIE Dataset Quality API",
    version="0.3.0",
    description=(
        "HTTP-сервис для оценки готовности датасета к обучению модели. "
        "Использует эвристики качества данных из EDA-ядра. "
        "Включает структурированное логирование и метрики."
    ),
    docs_url="/docs",
    redoc_url=None,
)

# ---------- Счетчики для метрик (Вариант F) ----------

request_stats = {
    "total_requests": 0,
    "total_latency_ms": 0.0,
    "endpoint_calls": {},
    "last_ok_for_model": None,
    "errors": 0,
}


# ---------- Модели запросов/ответов ----------

class QualityRequest(BaseModel):
    """Агрегированные признаки датасета."""
    n_rows: int = Field(..., ge=0, description="Число строк в датасете")
    n_cols: int = Field(..., ge=0, description="Число колонок")
    max_missing_share: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Максимальная доля пропусков среди всех колонок (0..1)",
    )
    numeric_cols: int = Field(
        ...,
        ge=0,
        description="Количество числовых колонок",
    )
    categorical_cols: int = Field(
        ...,
        ge=0,
        description="Количество категориальных колонок",
    )


class QualityResponse(BaseModel):
    """Ответ модели качества датасета."""
    ok_for_model: bool = Field(
        ...,
        description="True, если датасет считается достаточно качественным для обучения модели",
    )
    quality_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Интегральная оценка качества данных (0..1)",
    )
    message: str = Field(
        ...,
        description="Человекочитаемое пояснение решения",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Время обработки запроса на сервере, миллисекунды",
    )
    flags: dict[str, bool] | None = Field(
        default=None,
        description="Булевы флаги с подробностями",
    )
    dataset_shape: dict[str, int] | None = Field(
        default=None,
        description="Размеры датасета: {'n_rows': ..., 'n_cols': ...}",
    )


class QualityFlagsResponse(BaseModel):
    """Ответ с полным набором флагов качества датасета."""
    flags: dict = Field(
        ...,
        description="Полный набор флагов качества (булевы + списки проблемных колонок + метрики)",
    )
    quality_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Интегральная оценка качества данных (0..100)",
    )
    latency_ms: float = Field(
        ...,
        ge=0.0,
        description="Время обработки запроса на сервере, миллисекунды",
    )
    dataset_shape: dict[str, int] = Field(
        ...,
        description="Размеры датасета: {'n_rows': ..., 'n_cols': ...}",
    )


# ---------- Системный эндпоинт ----------

@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Простейший health-check сервиса."""
    return {
        "status": "ok",
        "service": "dataset-quality",
        "version": "0.3.0",
    }


# ---------- Заглушка /quality по агрегированным признакам ----------

@app.post("/quality", response_model=QualityResponse, tags=["quality"])
def quality(req: QualityRequest) -> QualityResponse:
    """
    Эндпоинт-заглушка, который принимает агрегированные признаки датасета
    и возвращает эвристическую оценку качества.
    """
    request_id = str(uuid.uuid4())
    start = perf_counter()

    # Обновляем статистику
    request_stats["total_requests"] += 1
    request_stats["endpoint_calls"]["quality"] = request_stats["endpoint_calls"].get("quality", 0) + 1

    # Базовый скор от 0 до 1
    score = 1.0

    # Чем больше пропусков, тем хуже
    score -= req.max_missing_share

    # Штраф за слишком маленький датасет
    if req.n_rows < 1000:
        score -= 0.2

    # Штраф за слишком широкий датасет
    if req.n_cols > 100:
        score -= 0.1

    # Штрафы за перекос по типам признаков
    if req.numeric_cols == 0 and req.categorical_cols > 0:
        score -= 0.1
    if req.categorical_cols == 0 and req.numeric_cols > 0:
        score -= 0.05

    # Нормируем скор в диапазон [0, 1]
    score = max(0.0, min(1.0, score))

    # Простое решение "ок / не ок"
    ok_for_model = score >= 0.7

    if ok_for_model:
        message = "Данных достаточно, модель можно обучать (по текущим эвристикам)."
    else:
        message = "Качество данных недостаточно, требуется доработка (по текущим эвристикам)."

    latency_ms = (perf_counter() - start) * 1000.0
    request_stats["total_latency_ms"] += latency_ms
    request_stats["last_ok_for_model"] = ok_for_model

    # Флаги для последующего логирования/аналитики
    flags = {
        "too_few_rows": req.n_rows < 1000,
        "too_many_columns": req.n_cols > 100,
        "too_many_missing": req.max_missing_share > 0.5,
        "no_numeric_columns": req.numeric_cols == 0,
        "no_categorical_columns": req.categorical_cols == 0,
    }

    # Структурированное логирование
    log_request(
        endpoint="quality",
        status=200,
        latency_ms=latency_ms,
        request_id=request_id,
        n_rows=req.n_rows,
        n_cols=req.n_cols,
        ok_for_model=ok_for_model,
        quality_score=score,
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags,
        dataset_shape={"n_rows": req.n_rows, "n_cols": req.n_cols},
    )


# ---------- /quality-from-csv: реальный CSV через EDA-логику ----------

@app.post(
    "/quality-from-csv",
    response_model=QualityResponse,
    tags=["quality"],
    summary="Оценка качества по CSV-файлу с использованием EDA-ядра",
)
async def quality_from_csv(file: UploadFile = File(...)) -> QualityResponse:
    """
    Эндпоинт, который принимает CSV-файл, запускает EDA-ядро
    и возвращает оценку качества данных.
    """
    request_id = str(uuid.uuid4())
    start = perf_counter()

    # Обновляем статистику
    request_stats["total_requests"] += 1
    request_stats["endpoint_calls"]["quality-from-csv"] = request_stats["endpoint_calls"].get("quality-from-csv", 0) + 1

    # Проверка типа файла
    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        request_stats["errors"] += 1
        log_request(
            endpoint="quality-from-csv",
            status=400,
            latency_ms=(perf_counter() - start) * 1000.0,
            request_id=request_id,
            error="Invalid content type",
        )
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл (content-type text/csv).")

    try:
        # Читаем CSV
        df = pd.read_csv(file.file)
    except Exception as exc:
        request_stats["errors"] += 1
        latency_ms = (perf_counter() - start) * 1000.0
        log_request(
            endpoint="quality-from-csv",
            status=400,
            latency_ms=latency_ms,
            request_id=request_id,
            error=str(exc),
        )
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        request_stats["errors"] += 1
        log_request(
            endpoint="quality-from-csv",
            status=400,
            latency_ms=(perf_counter() - start) * 1000.0,
            request_id=request_id,
            error="Empty DataFrame",
        )
        raise HTTPException(status_code=400, detail="CSV-файл не содержит данных (пустой DataFrame).")

    # Используем EDA-ядро из HW03
    stats = get_basic_stats(df)
    quality_flags = compute_quality_flags(df)

    # Качественный скор из compute_quality_flags (0-100), нормируем в [0,1]
    score = quality_flags.get("quality_score", 0) / 100.0
    score = max(0.0, min(1.0, score))

    ok_for_model = score >= 0.7

    if ok_for_model:
        message = "CSV выглядит достаточно качественным для обучения модели (по текущим эвристикам)."
    else:
        message = "CSV требует доработки перед обучением модели (по текущим эвристикам)."

    latency_ms = (perf_counter() - start) * 1000.0
    request_stats["total_latency_ms"] += latency_ms
    request_stats["last_ok_for_model"] = ok_for_model

    # Собираем булевы флаги из quality_flags
    flags_bool: dict[str, bool] = {
        "has_high_missing": bool(quality_flags.get("has_high_missing", False)),
        "has_duplicates": bool(quality_flags.get("has_duplicates", False)),
        "has_constant_columns": bool(quality_flags.get("has_constant_columns", False)),
        "has_high_cardinality_categoricals": bool(quality_flags.get("has_high_cardinality_categoricals", False)),
        "has_many_zero_values": bool(quality_flags.get("has_many_zero_values", False)),
    }

    # Размеры датасета
    n_rows = stats["n_rows"]
    n_cols = stats["n_cols"]

    # Структурированное логирование
    log_request(
        endpoint="quality-from-csv",
        status=200,
        latency_ms=latency_ms,
        request_id=request_id,
        filename=file.filename,
        n_rows=n_rows,
        n_cols=n_cols,
        ok_for_model=ok_for_model,
        quality_score=score,
    )

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags_bool,
        dataset_shape={"n_rows": n_rows, "n_cols": n_cols},
    )


# ---------- Новый эндпоинт для HW04: полные флаги качества ----------

@app.post(
    "/quality-flags-from-csv",
    response_model=QualityFlagsResponse,
    tags=["quality"],
    summary="Получить полный набор флагов качества из CSV-файла (HW04)",
)
async def quality_flags_from_csv(file: UploadFile = File(...)) -> QualityFlagsResponse:
    """
    Эндпоинт для получения полного набора флагов качества датасета.

    Возвращает все эвристики из HW03:
    - has_high_missing: есть ли колонки с высокой долей пропусков
    - has_duplicates: есть ли дубликаты строк
    - has_constant_columns: есть ли константные колонки
    - has_high_cardinality_categoricals: есть ли категориальные колонки с высокой кардинальностью
    - has_many_zero_values: есть ли числовые колонки с большим количеством нулей

    А также списки проблемных колонок и дополнительные метрики.
    """
    request_id = str(uuid.uuid4())
    start = perf_counter()

    # Обновляем статистику
    request_stats["total_requests"] += 1
    request_stats["endpoint_calls"]["quality-flags-from-csv"] = request_stats["endpoint_calls"].get("quality-flags-from-csv", 0) + 1

    # Проверка типа файла
    if file.content_type not in ("text/csv", "application/vnd.ms-excel", "application/octet-stream"):
        request_stats["errors"] += 1
        log_request(
            endpoint="quality-flags-from-csv",
            status=400,
            latency_ms=(perf_counter() - start) * 1000.0,
            request_id=request_id,
            error="Invalid content type",
        )
        raise HTTPException(status_code=400, detail="Ожидается CSV-файл (content-type text/csv).")

    try:
        # Читаем CSV
        df = pd.read_csv(file.file)
    except Exception as exc:
        request_stats["errors"] += 1
        latency_ms = (perf_counter() - start) * 1000.0
        log_request(
            endpoint="quality-flags-from-csv",
            status=400,
            latency_ms=latency_ms,
            request_id=request_id,
            error=str(exc),
        )
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать CSV: {exc}")

    if df.empty:
        request_stats["errors"] += 1
        log_request(
            endpoint="quality-flags-from-csv",
            status=400,
            latency_ms=(perf_counter() - start) * 1000.0,
            request_id=request_id,
            error="Empty DataFrame",
        )
        raise HTTPException(status_code=400, detail="CSV-файл не содержит данных (пустой DataFrame).")

    # Получаем все флаги качества из EDA-ядра HW03
    quality_flags = compute_quality_flags(df)

    # Конвертируем numpy типы в нативные Python типы
    quality_flags = convert_to_native_types(quality_flags)

    # Извлекаем quality_score
    quality_score = quality_flags.pop("quality_score", 0)

    latency_ms = (perf_counter() - start) * 1000.0
    request_stats["total_latency_ms"] += latency_ms

    # Размеры датасета
    n_rows, n_cols = df.shape

    # Структурированное логирование
    log_request(
        endpoint="quality-flags-from-csv",
        status=200,
        latency_ms=latency_ms,
        request_id=request_id,
        filename=file.filename,
        n_rows=int(n_rows),
        n_cols=int(n_cols),
        quality_score=quality_score,
    )

    return QualityFlagsResponse(
        flags=quality_flags,
        quality_score=quality_score,
        latency_ms=latency_ms,
        dataset_shape={"n_rows": int(n_rows), "n_cols": int(n_cols)},
    )


# ---------- Вариант F: Эндпоинт /metrics ----------

@app.get("/metrics", tags=["system"])
def metrics() -> dict:
    """
    Возвращает статистику по работе сервиса.

    Включает:
    - total_requests: общее количество запросов
    - avg_latency_ms: среднее время обработки запроса
    - endpoint_calls: количество вызовов по каждому эндпоинту
    - last_ok_for_model: последнее значение флага ok_for_model
    - errors: количество ошибок
    """
    avg_latency = 0.0
    if request_stats["total_requests"] > 0:
        avg_latency = request_stats["total_latency_ms"] / request_stats["total_requests"]

    return {
        "total_requests": request_stats["total_requests"],
        "avg_latency_ms": round(avg_latency, 2),
        "endpoint_calls": request_stats["endpoint_calls"],
        "last_ok_for_model": request_stats["last_ok_for_model"],
        "errors": request_stats["errors"],
    }
