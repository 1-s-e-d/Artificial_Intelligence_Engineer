from __future__ import annotations

import json
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Dict

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .core import compute_quality_flags, missing_table, summarize_dataset

app = FastAPI(
    title="AIE Dataset Quality API",
    version="0.3.0",
    description=(
        "HTTP-сервис-заглушка для оценки готовности датасета к обучению модели. "
        "Использует простые эвристики качества данных вместо настоящей ML-модели."
    ),
    docs_url="/docs",
    redoc_url=None,
)

# ---------------- Логирование ----------------

LOG_DIR = Path(__file__).parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "api.log"


def write_log(log_data: dict[str, Any]) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_data, ensure_ascii=False) + "\n")


# ---------------- Метрики сервиса ----------------

metrics_store: dict[str, Any] = {
    "total_requests": 0,
    "total_latency_ms": 0.0,
    "avg_latency_ms": 0.0,
    "endpoint_calls": defaultdict(int),
    "last_ok_for_model": None,
    "errors": 0,
}


def update_metrics(
        endpoint: str,
        latency_ms: float,
        ok_for_model: bool | None,
        error: bool = False,
) -> None:
    metrics_store["total_requests"] += 1
    metrics_store["total_latency_ms"] += latency_ms
    if metrics_store["total_requests"] > 0:
        metrics_store["avg_latency_ms"] = (
                metrics_store["total_latency_ms"] / metrics_store["total_requests"]
        )
    metrics_store["endpoint_calls"][endpoint] += 1
    if ok_for_model is not None:
        metrics_store["last_ok_for_model"] = bool(ok_for_model)
    if error:
        metrics_store["errors"] += 1


# ---------- Модели запросов/ответов ----------


class QualityRequest(BaseModel):
    """Агрегированные признаки датасета – 'фичи' для заглушки модели."""

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
    """Ответ заглушки модели качества датасета."""

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
        description="Булевы флаги с подробностями (например, too_few_rows, too_many_missing)",
    )
    dataset_shape: dict[str, int] | None = Field(
        default=None,
        description="Размеры датасета: {'n_rows': ..., 'n_cols': ...}, если известны",
    )


# ---------- Системные эндпоинты ----------


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Простейший health-check сервиса."""
    return {
        "status": "ok",
        "service": "dataset-quality",
        "version": "0.3.0",
    }


@app.get("/metrics", tags=["system"])
def get_metrics() -> Dict[str, Any]:
    """
    Метрики работы сервиса:
    - total_requests
    - avg_latency_ms
    - endpoint_calls
    - last_ok_for_model
    - errors
    """
    return {
        "total_requests": metrics_store["total_requests"],
        "avg_latency_ms": round(float(metrics_store["avg_latency_ms"]), 2),
        "endpoint_calls": dict(metrics_store["endpoint_calls"]),
        "last_ok_for_model": metrics_store["last_ok_for_model"],
        "errors": metrics_store["errors"],
    }


# ---------- Заглушка /quality по агрегированным признакам ----------


@app.post("/quality", response_model=QualityResponse, tags=["quality"])
def quality(req: QualityRequest) -> QualityResponse:
    """
    Эндпоинт-заглушка, который принимает агрегированные признаки датасета
    и возвращает эвристическую оценку качества.
    """
    start = perf_counter()
    request_id = str(uuid.uuid4())

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

    # Флаги
    flags = {
        "too_few_rows": req.n_rows < 1000,
        "too_many_columns": req.n_cols > 100,
        "too_many_missing": req.max_missing_share > 0.5,
        "no_numeric_columns": req.numeric_cols == 0,
        "no_categorical_columns": req.categorical_cols == 0,
    }

    print(
        f"[quality] n_rows={req.n_rows} n_cols={req.n_cols} "
        f"max_missing_share={req.max_missing_share:.3f} "
        f"score={score:.3f} latency_ms={latency_ms:.1f} ms"
    )

    # Логирование
    write_log({
        "timestamp": datetime.now().isoformat(),
        "request_id": request_id,
        "endpoint": "quality",
        "status": 200,
        "latency_ms": round(latency_ms, 2),
        "n_rows": req.n_rows,
        "n_cols": req.n_cols,
        "quality_score": int(round(score * 100)),
    })

    update_metrics("quality", latency_ms, ok_for_model)

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags,
        dataset_shape={"n_rows": req.n_rows, "n_cols": req.n_cols},
    )


# ---------- /quality-from-csv: реальный CSV через нашу EDA-логику ----------


@app.post(
    "/quality-from-csv",
    response_model=QualityResponse,
    tags=["quality"],
    summary="Оценка качества по CSV-файлу с использованием EDA-ядра",
)
async def quality_from_csv(file: UploadFile = File(...)) -> QualityResponse:
    """
    Эндпоинт, который принимает CSV-файл, запускает EDA-ядро
    (summarize_dataset + missing_table + compute_quality_flags)
    и возвращает оценку качества данных.
    """
    start = perf_counter()
    request_id = str(uuid.uuid4())

    if file.content_type not in (
            "text/csv",
            "application/vnd.ms-excel",
            "application/octet-stream",
    ):
        update_metrics("quality-from-csv", 0.0, None, error=True)
        raise HTTPException(
            status_code=400,
            detail="Ожидается CSV-файл (content-type text/csv).",
        )

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:  # noqa: BLE001
        update_metrics("quality-from-csv", 0.0, None, error=True)
        raise HTTPException(
            status_code=400,
            detail=f"Не удалось прочитать CSV: {exc}",
        )

    if df.empty:
        update_metrics("quality-from-csv", 0.0, None, error=True)
        raise HTTPException(
            status_code=400,
            detail="CSV-файл не содержит данных (пустой DataFrame).",
        )

    # Используем EDA-ядро из S03
    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags_all = compute_quality_flags(df)

    # compute_quality_flags возвращает score в диапазоне 0-100
    score_0_100 = int(flags_all.get("quality_score", 0))
    score_0_100 = max(0, min(100, score_0_100))

    # Для QualityResponse нужен диапазон 0.0-1.0
    score = score_0_100 / 100.0

    ok_for_model = score >= 0.7
    if ok_for_model:
        message = (
            "CSV выглядит достаточно качественным для обучения модели "
            "(по текущим эвристикам)."
        )
    else:
        message = (
            "CSV требует доработки перед обучением модели "
            "(по текущим эвристикам)."
        )

    latency_ms = (perf_counter() - start) * 1000.0

    # Оставляем только булевы флаги для компактности
    flags_bool: dict[str, bool] = {
        key: bool(value)
        for key, value in flags_all.items()
        if isinstance(value, bool)
    }

    # Размеры датасета
    try:
        n_rows = int(getattr(summary, "n_rows"))
        n_cols = int(getattr(summary, "n_cols"))
    except AttributeError:
        n_rows = int(df.shape[0])
        n_cols = int(df.shape[1])

    print(
        f"[quality-from-csv] filename={file.filename!r} "
        f"n_rows={n_rows} n_cols={n_cols} score={score:.3f} "
        f"latency_ms={latency_ms:.1f} ms"
    )

    # Логирование
    write_log({
        "timestamp": datetime.now().isoformat(),
        "request_id": request_id,
        "endpoint": "quality-from-csv",
        "status": 200,
        "latency_ms": round(latency_ms, 2),
        "filename": file.filename,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "quality_score": score_0_100,
    })

    update_metrics("quality-from-csv", latency_ms, ok_for_model)

    return QualityResponse(
        ok_for_model=ok_for_model,
        quality_score=score,
        message=message,
        latency_ms=latency_ms,
        flags=flags_bool,
        dataset_shape={"n_rows": n_rows, "n_cols": n_cols},
    )


# ---------- /quality-flags-from-csv: полный набор флагов ----------


@app.post(
    "/quality-flags-from-csv",
    tags=["quality"],
    summary="Полный набор флагов качества по CSV-файлу",
)
async def quality_flags_from_csv(
        file: UploadFile = File(...),
) -> Dict[str, Any]:
    """
    Эндпоинт, который принимает CSV-файл и возвращает полный набор
    флагов качества датасета:
      - high_missing_columns
      - duplicate_count
      - constant_columns
      - high_cardinality_columns
      - high_zero_columns
      - zero_shares
    а также интегральный quality_score и размеры датасета.
    """
    start = perf_counter()
    request_id = str(uuid.uuid4())

    if file.content_type not in (
            "text/csv",
            "application/vnd.ms-excel",
            "application/octet-stream",
    ):
        update_metrics("quality-flags-from-csv", 0.0, None, error=True)
        raise HTTPException(
            status_code=400,
            detail="Ожидается CSV-файл (content-type text/csv).",
        )

    try:
        df = pd.read_csv(file.file)
    except Exception as exc:  # noqa: BLE001
        update_metrics("quality-flags-from-csv", 0.0, None, error=True)
        raise HTTPException(
            status_code=400,
            detail=f"Не удалось прочитать CSV: {exc}",
        )

    if df.empty:
        update_metrics("quality-flags-from-csv", 0.0, None, error=True)
        raise HTTPException(
            status_code=400,
            detail="CSV-файл не содержит данных (пустой DataFrame).",
        )

    summary = summarize_dataset(df)
    missing_df = missing_table(df)
    flags_all = compute_quality_flags(df)

    # compute_quality_flags возвращает score в диапазоне 0-100
    quality_score = int(flags_all.get("quality_score", 0))
    quality_score = max(0, min(100, quality_score))

    # ---- детальные флаги ----

    # колонки с высокой долей пропусков (порог 30%, как в compute_quality_flags)
    high_missing_columns: list[str] = []
    if not missing_df.empty and "missing_percent" in missing_df.columns:
        for idx, row in missing_df.iterrows():
            col = row["column"]
            pct = row["missing_percent"]
            if pct > 30.0:  # ✅ 30% порог (0.3 * 100)
                high_missing_columns.append(str(col))

    # дубликаты
    duplicate_count = int(df.duplicated().sum())
    has_duplicates = duplicate_count > 0

    # константные колонки
    constant_columns = [
        col for col in df.columns if df[col].nunique(dropna=False) <= 1
    ]
    has_constant_columns = len(constant_columns) > 0

    # высококардинальные категориальные колонки
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns
    high_cardinality_columns = [
        col for col in categorical_cols if df[col].nunique(dropna=True) > 50
    ]
    has_high_cardinality_categoricals = len(high_cardinality_columns) > 0

    # нули по числовым колонкам
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    zero_shares: Dict[str, float] = {}
    high_zero_columns: list[str] = []
    for col in numeric_cols:
        share = (
            float((df[col] == 0).sum()) / float(len(df))
            if len(df) > 0
            else 0.0
        )
        zero_shares[str(col)] = share
        if share > 0.5:
            high_zero_columns.append(str(col))
    has_many_zero_values = len(high_zero_columns) > 0

    latency_ms = (perf_counter() - start) * 1000.0
    n_rows, n_cols = int(df.shape[0]), int(df.shape[1])

    flags: Dict[str, Any] = {
        "has_high_missing": len(high_missing_columns) > 0,
        "high_missing_columns": high_missing_columns,
        "has_duplicates": has_duplicates,
        "duplicate_count": duplicate_count,
        "has_constant_columns": has_constant_columns,
        "constant_columns": constant_columns,
        "has_high_cardinality_categoricals": has_high_cardinality_categoricals,
        "high_cardinality_columns": high_cardinality_columns,
        "has_many_zero_values": has_many_zero_values,
        "high_zero_columns": high_zero_columns,
        "zero_shares": {k: float(v) for k, v in zero_shares.items()},
    }

    print(
        f"[quality-flags-from-csv] filename={file.filename!r} "
        f"n_rows={n_rows} n_cols={n_cols} quality_score={quality_score} "
        f"latency_ms={latency_ms:.1f} ms"
    )

    # Логирование
    write_log({
        "timestamp": datetime.now().isoformat(),
        "request_id": request_id,
        "endpoint": "quality-flags-from-csv",
        "status": 200,
        "latency_ms": round(latency_ms, 2),
        "filename": file.filename,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "quality_score": quality_score,
    })

    # ok_for_model по порогу 70
    ok_for_model = quality_score >= 70
    update_metrics("quality-flags-from-csv", latency_ms, ok_for_model)

    return {
        "flags": flags,
        "quality_score": quality_score,
        "latency_ms": latency_ms,
        "dataset_shape": {"n_rows": n_rows, "n_cols": n_cols},
    }
