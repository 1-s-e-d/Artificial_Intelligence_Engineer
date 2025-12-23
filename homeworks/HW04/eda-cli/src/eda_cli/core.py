"""Основная логика анализа данных."""

import pandas as pd
from pathlib import Path


def load_csv(filepath: str, sep: str = ",", encoding: str = "utf-8") -> pd.DataFrame:
    """Загрузка CSV файла в DataFrame."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {filepath}")
    df = pd.read_csv(path, sep=sep, encoding=encoding)
    return df


def get_basic_stats(df: pd.DataFrame) -> dict:
    """Базовая статистика по датасету."""
    stats = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 3),
    }
    return stats


def get_missing_info(df: pd.DataFrame) -> dict:
    """Информация о пропущенных значениях."""
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)

    missing_info = {}
    for col in df.columns:
        if missing_counts[col] > 0:
            missing_info[col] = {
                "count": int(missing_counts[col]),
                "percent": float(missing_pct[col])
            }

    return {
        "total_missing": int(missing_counts.sum()),
        "columns_with_missing": len(missing_info),
        "details": missing_info
    }


def get_numeric_summary(df: pd.DataFrame) -> dict:
    """Статистика по числовым колонкам."""
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if not numeric_cols:
        return {"numeric_columns": [], "stats": {}}

    desc = df[numeric_cols].describe().to_dict()
    return {
        "numeric_columns": numeric_cols,
        "stats": desc
    }


def get_categorical_summary(df: pd.DataFrame, top_k: int = 5) -> dict:
    """Статистика по категориальным колонкам."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        return {"categorical_columns": [], "stats": {}}

    cat_stats = {}
    for col in cat_cols:
        value_counts = df[col].value_counts()
        cat_stats[col] = {
            "unique_count": int(df[col].nunique()),
            "top_values": value_counts.head(top_k).to_dict(),
            "null_count": int(df[col].isnull().sum())
        }

    return {
        "categorical_columns": cat_cols,
        "stats": cat_stats
    }


def summarize_dataset(df: pd.DataFrame) -> dict:
    """
    Создаёт полную сводку датасета (для использования в API).

    Возвращает словарь с базовой статистикой, информацией о пропусках,
    числовых и категориальных признаках.
    """
    stats = get_basic_stats(df)
    missing = get_missing_info(df)
    numeric = get_numeric_summary(df)
    categorical = get_categorical_summary(df)

    return {
        "basic_stats": stats,
        "missing_info": missing,
        "numeric": numeric,
        "categorical": categorical,
        "n_rows": len(df),
        "n_cols": len(df.columns)
    }


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Возвращает таблицу с информацией о пропусках по каждой колонке.

    Возвращаемый DataFrame содержит колонки:
    - column: название колонки
    - missing_count: количество пропусков
    - missing_percent: процент пропусков
    """
    n_rows = len(df)
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / n_rows * 100).round(2)

    missing_data = []
    for col in df.columns:
        missing_data.append({
            "column": col,
            "missing_count": int(missing_counts[col]),
            "missing_percent": float(missing_pct[col])
        })

    return pd.DataFrame(missing_data)


def compute_quality_flags(
    df: pd.DataFrame,
    missing_threshold: float = 0.3,
    high_cardinality_threshold: int = 50,
    zero_threshold: float = 0.5
) -> dict:
    """
    Вычисление флагов качества данных.

    Параметры:
        df: DataFrame для анализа
        missing_threshold: порог доли пропусков (по умолчанию 30%)
        high_cardinality_threshold: порог уникальных значений для категориальных
        zero_threshold: порог доли нулей в числовых колонках

    Возвращает:
        dict с флагами качества и интегральным quality_score (0-100)
    """
    n_rows = len(df)
    flags = {}

    # Проверка на пропуски
    missing_shares = df.isnull().sum() / n_rows
    cols_high_missing = [col for col, share in missing_shares.items() if share > missing_threshold]
    flags["has_high_missing"] = len(cols_high_missing) > 0
    flags["high_missing_columns"] = cols_high_missing

    # Проверка на дубликаты строк
    dup_count = df.duplicated().sum()
    flags["has_duplicates"] = dup_count > 0
    flags["duplicate_count"] = int(dup_count)

    # Проверка на константные колонки
    constant_cols = []
    for col in df.columns:
        if df[col].nunique(dropna=True) <= 1:
            constant_cols.append(col)
    flags["has_constant_columns"] = len(constant_cols) > 0
    flags["constant_columns"] = constant_cols

    # Проверка на высокую кардинальность категориальных признаков
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    high_card_cols = []
    for col in cat_cols:
        unique_vals = df[col].nunique()
        if unique_vals > high_cardinality_threshold:
            high_card_cols.append(col)
    flags["has_high_cardinality_categoricals"] = len(high_card_cols) > 0
    flags["high_cardinality_columns"] = high_card_cols

    # Проверка на много нулевых значений в числовых колонках
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    high_zero_cols = []
    zero_shares = {}
    for col in numeric_cols:
        zero_share = (df[col] == 0).sum() / n_rows
        zero_shares[col] = round(float(zero_share), 4)
        if zero_share > zero_threshold:
            high_zero_cols.append(col)
    flags["has_many_zero_values"] = len(high_zero_cols) > 0
    flags["high_zero_columns"] = high_zero_cols
    flags["zero_shares"] = zero_shares

    # Расчет интегрального quality_score (0-100)
    penalties = 0
    if flags["has_high_missing"]:
        penalties += 20
    if flags["has_duplicates"]:
        penalties += 15
    if flags["has_constant_columns"]:
        penalties += 10
    if flags["has_high_cardinality_categoricals"]:
        penalties += 10
    if flags["has_many_zero_values"]:
        penalties += 5

    flags["quality_score"] = max(0, 100 - penalties)

    return flags


def get_problematic_columns(df: pd.DataFrame, min_missing_share: float = 0.1) -> list:
    """Возвращает список проблемных колонок по заданному порогу пропусков."""
    n_rows = len(df)
    problematic = []

    for col in df.columns:
        missing_share = df[col].isnull().sum() / n_rows
        if missing_share >= min_missing_share:
            problematic.append({
                "column": col,
                "missing_share": round(missing_share, 4),
                "missing_count": int(df[col].isnull().sum())
            })

    return sorted(problematic, key=lambda x: x["missing_share"], reverse=True)
