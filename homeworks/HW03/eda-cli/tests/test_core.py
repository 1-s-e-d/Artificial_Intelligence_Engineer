"""Тесты для модуля core."""

import pytest
import pandas as pd
from eda_cli import core


class TestLoadCsv:
    """Тесты загрузки CSV."""

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            core.load_csv("nonexistent_file.csv")


class TestBasicStats:
    """Тесты базовой статистики."""

    def test_basic_stats_shape(self):
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"]
        })
        stats = core.get_basic_stats(df)

        assert stats["n_rows"] == 3
        assert stats["n_cols"] == 2
        assert "a" in stats["columns"]
        assert "b" in stats["columns"]


class TestMissingInfo:
    """Тесты информации о пропусках."""

    def test_no_missing(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        missing = core.get_missing_info(df)

        assert missing["total_missing"] == 0
        assert missing["columns_with_missing"] == 0

    def test_with_missing(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 6]})
        missing = core.get_missing_info(df)

        assert missing["total_missing"] == 3
        assert missing["columns_with_missing"] == 2


class TestQualityFlags:
    """Тесты флагов качества данных."""

    def test_has_duplicates(self):
        df = pd.DataFrame({
            "a": [1, 1, 2],
            "b": ["x", "x", "y"]
        })
        flags = core.compute_quality_flags(df)

        assert flags["has_duplicates"] == True
        assert flags["duplicate_count"] == 1

    def test_no_duplicates(self):
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"]
        })
        flags = core.compute_quality_flags(df)

        assert flags["has_duplicates"] == False

    def test_constant_columns_detected(self):
        """Тест обнаружения константных колонок."""
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "constant_col": ["same", "same", "same", "same", "same"],
            "normal_col": ["a", "b", "c", "d", "e"]
        })
        flags = core.compute_quality_flags(df)

        assert flags["has_constant_columns"] == True
        assert "constant_col" in flags["constant_columns"]
        assert "normal_col" not in flags["constant_columns"]

    def test_no_constant_columns(self):
        """Тест когда нет константных колонок."""
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"]
        })
        flags = core.compute_quality_flags(df)

        assert flags["has_constant_columns"] == False
        assert len(flags["constant_columns"]) == 0

    def test_high_cardinality_detected(self):
        """Тест обнаружения высокой кардинальности."""
        # Создаем датафрейм с категориальной колонкой, где много уникальных значений
        df = pd.DataFrame({
            "id": range(100),
            "category": [f"cat_{i}" for i in range(100)]  # 100 уникальных
        })

        flags = core.compute_quality_flags(df, high_cardinality_threshold=50)

        assert flags["has_high_cardinality_categoricals"] == True
        assert "category" in flags["high_cardinality_columns"]

    def test_low_cardinality_ok(self):
        """Тест когда кардинальность в пределах нормы."""
        df = pd.DataFrame({
            "id": range(100),
            "category": ["A", "B", "C"] * 33 + ["A"]
        })

        flags = core.compute_quality_flags(df, high_cardinality_threshold=50)

        assert flags["has_high_cardinality_categoricals"] == False

    def test_many_zeros_detected(self):
        """Тест обнаружения большой доли нулей."""
        df = pd.DataFrame({
            "values": [0, 0, 0, 0, 0, 0, 0, 0, 1, 2]  # 80% нулей
        })

        flags = core.compute_quality_flags(df, zero_threshold=0.5)

        assert flags["has_many_zero_values"] == True
        assert "values" in flags["high_zero_columns"]

    def test_quality_score_calculation(self):
        """Тест расчета quality_score."""
        # Чистый датасет
        df_clean = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"]
        })
        flags_clean = core.compute_quality_flags(df_clean)
        assert flags_clean["quality_score"] == 100

        # Датасет с дубликатами
        df_dups = pd.DataFrame({
            "a": [1, 1, 3],
            "b": ["x", "x", "z"]
        })
        flags_dups = core.compute_quality_flags(df_dups)
        assert flags_dups["quality_score"] < 100


class TestProblematicColumns:
    """Тесты для проблемных колонок."""

    def test_problematic_columns_found(self):
        df = pd.DataFrame({
            "good": [1, 2, 3, 4, 5],
            "bad": [None, None, None, 4, 5]  # 60% пропусков
        })

        problematic = core.get_problematic_columns(df, min_missing_share=0.5)

        assert len(problematic) == 1
        assert problematic[0]["column"] == "bad"

    def test_no_problematic_columns(self):
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6]
        })

        problematic = core.get_problematic_columns(df, min_missing_share=0.1)

        assert len(problematic) == 0
