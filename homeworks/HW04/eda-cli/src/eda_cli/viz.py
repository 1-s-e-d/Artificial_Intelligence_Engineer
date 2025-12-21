"""Модуль визуализации для EDA."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def save_histograms(
        df: pd.DataFrame,
        out_dir: Path,
        max_columns: int = 6,
        filename: str = "histograms.png"
) -> str:
    """
    Построение гистограмм для числовых колонок.

    Параметры:
        df: DataFrame
        out_dir: директория для сохранения
        max_columns: максимальное число колонок для отображения
        filename: имя файла

    Возвращает путь к сохраненному файлу.
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if not numeric_cols:
        return ""

    cols_to_plot = numeric_cols[:max_columns]
    n_cols = len(cols_to_plot)

    ncols_grid = min(3, n_cols)
    nrows_grid = (n_cols + ncols_grid - 1) // ncols_grid

    fig, axes = plt.subplots(nrows_grid, ncols_grid, figsize=(4 * ncols_grid, 3 * nrows_grid))

    if n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, col in enumerate(cols_to_plot):
        ax = axes[idx]
        df[col].dropna().hist(ax=ax, bins=20, edgecolor="black", alpha=0.7)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Частота")

    # Скрыть пустые subplot'ы
    for idx in range(n_cols, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    out_path = out_dir / filename
    plt.savefig(out_path, dpi=100)
    plt.close(fig)

    return str(out_path)


def save_missing_bar(df: pd.DataFrame, out_dir: Path, filename: str = "missing_bar.png") -> str:
    """Столбчатая диаграмма пропусков по колонкам."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) == 0:
        return ""

    fig, ax = plt.subplots(figsize=(8, max(4, len(missing) * 0.4)))

    missing.plot(kind="barh", ax=ax, color="coral", edgecolor="black")
    ax.set_xlabel("Количество пропусков")
    ax.set_ylabel("Колонка")
    ax.set_title("Пропуски по колонкам")

    plt.tight_layout()

    out_path = out_dir / filename
    plt.savefig(out_path, dpi=100)
    plt.close(fig)

    return str(out_path)


def save_boxplots(
        df: pd.DataFrame,
        out_dir: Path,
        max_columns: int = 6,
        filename: str = "boxplots.png"
) -> str:
    """
    Построение boxplot для числовых колонок.
    Дополнительная визуализация (Вариант C).
    """
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if not numeric_cols:
        return ""

    cols_to_plot = numeric_cols[:max_columns]
    n_cols = len(cols_to_plot)

    fig, ax = plt.subplots(figsize=(max(8, n_cols * 1.5), 6))

    data_to_plot = [df[col].dropna().values for col in cols_to_plot]

    bp = ax.boxplot(data_to_plot, labels=cols_to_plot, patch_artist=True)

    colors = plt.cm.Set3.colors
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax.set_ylabel("Значения")
    ax.set_title("Boxplot числовых признаков")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    out_path = out_dir / filename
    plt.savefig(out_path, dpi=100)
    plt.close(fig)

    return str(out_path)


def save_category_bar(
        df: pd.DataFrame,
        column: str,
        out_dir: Path,
        top_n: int = 10,
        filename: str = None
) -> str:
    """Столбчатая диаграмма для категориального признака."""
    if column not in df.columns:
        return ""

    value_counts = df[column].value_counts().head(top_n)

    if len(value_counts) == 0:
        return ""

    fig, ax = plt.subplots(figsize=(8, 5))

    value_counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_xlabel(column)
    ax.set_ylabel("Количество")
    ax.set_title(f"Top-{top_n} значений: {column}")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    if filename is None:
        filename = f"category_{column}.png"

    out_path = out_dir / filename
    plt.savefig(out_path, dpi=100)
    plt.close(fig)

    return str(out_path)
