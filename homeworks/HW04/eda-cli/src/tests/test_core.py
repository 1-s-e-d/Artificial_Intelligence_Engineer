"""
Модульные тесты для модуля core.

Данный модуль содержит набор тестов для проверки корректности работы функций
из модуля eda_cli.core, включая загрузку данных, вычисление статистики,
анализ качества данных и определение проблемных колонок.
"""

import pytest
import pandas as pd
from eda_cli import core


class TestLoadCsv:
    """Набор тестов для функции загрузки CSV-файлов."""

    def test_load_nonexistent_file(self):
        """Проверка обработки ошибки при загрузке несуществующего файла."""
        with pytest.raises(FileNotFoundError):
            core.load_csv("nonexistent_file.csv")


class TestBasicStats:
    """Набор тестов для функции вычисления базовой статистики датасета."""

    def test_basic_stats_shape(self):
        """
        Проверка корректности вычисления размерности датасета.

        Тест проверяет, что функция get_basic_stats правильно определяет
        количество строк, колонок и список названий колонок.
        """
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
    """Набор тестов для функции анализа пропущенных значений."""

    def test_no_missing(self):
        """
        Проверка корректности работы при отсутствии пропусков.

        Тест проверяет, что функция корректно определяет отсутствие
        пропущенных значений в датасете.
        """
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        missing = core.get_missing_info(df)

        assert missing["total_missing"] == 0
        assert missing["columns_with_missing"] == 0

    def test_with_missing(self):
        """
        Проверка корректности подсчёта пропущенных значений.

        Тест проверяет правильность определения общего количества пропусков
        и количества колонок, содержащих пропущенные значения.
        """
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, 6]})
        missing = core.get_missing_info(df)

        assert missing["total_missing"] == 3
        assert missing["columns_with_missing"] == 2


class TestQualityFlags:
    """Набор тестов для функции вычисления флагов качества данных."""

    def test_has_duplicates(self):
        """
        Проверка обнаружения дублирующихся строк.

        Тест проверяет, что функция корректно устанавливает флаг наличия
        дубликатов и правильно подсчитывает их количество.
        """
        df = pd.DataFrame({
            "a": [1, 1, 2],
            "b": ["x", "x", "y"]
        })
        flags = core.compute_quality_flags(df)

        assert flags["has_duplicates"] == True
        assert flags["duplicate_count"] == 1

    def test_no_duplicates(self):
        """
        Проверка корректности работы при отсутствии дубликатов.

        Тест проверяет, что функция корректно определяет отсутствие
        дублирующихся строк в датасете.
        """
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"]
        })
        flags = core.compute_quality_flags(df)

        assert flags["has_duplicates"] == False

    def test_constant_columns_detected(self):
        """
        Проверка обнаружения колонок с константными значениями.

        Тест проверяет способность функции выявлять колонки, в которых
        все значения идентичны, и корректно их идентифицировать.
        """
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
        """
        Проверка корректности работы при отсутствии константных колонок.

        Тест проверяет, что функция корректно определяет отсутствие
        колонок с одинаковыми значениями во всех строках.
        """
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"]
        })
        flags = core.compute_quality_flags(df)

        assert flags["has_constant_columns"] == False
        assert len(flags["constant_columns"]) == 0

    def test_high_cardinality_detected(self):
        """
        Проверка обнаружения категориальных признаков с высокой кардинальностью.

        Тест проверяет способность функции выявлять категориальные колонки
        с чрезмерно большим числом уникальных значений относительно заданного порога.
        """
        df = pd.DataFrame({
            "id": range(100),
            "category": [f"cat_{i}" for i in range(100)]
        })

        flags = core.compute_quality_flags(df, high_cardinality_threshold=50)

        assert flags["has_high_cardinality_categoricals"] == True
        assert "category" in flags["high_cardinality_columns"]

    def test_low_cardinality_ok(self):
        """
        Проверка корректности работы с колонками нормальной кардинальности.

        Тест проверяет, что функция не классифицирует как проблемные
        категориальные колонки с допустимым числом уникальных значений.
        """
        df = pd.DataFrame({
            "id": range(100),
            "category": ["A", "B", "C"] * 33 + ["A"]
        })

        flags = core.compute_quality_flags(df, high_cardinality_threshold=50)

        assert flags["has_high_cardinality_categoricals"] == False

    def test_many_zeros_detected(self):
        """
        Проверка обнаружения колонок с избыточным количеством нулевых значений.

        Тест проверяет способность функции выявлять числовые колонки,
        в которых доля нулевых значений превышает установленный порог.
        """
        df = pd.DataFrame({
            "values": [0, 0, 0, 0, 0, 0, 0, 0, 1, 2]
        })

        flags = core.compute_quality_flags(df, zero_threshold=0.5)

        assert flags["has_many_zero_values"] == True
        assert "values" in flags["high_zero_columns"]

    def test_quality_score_calculation(self):
        """
        Проверка корректности расчёта интегрального показателя качества данных.

        Тест проверяет, что функция правильно вычисляет quality_score:
        - присваивает максимальную оценку (100) для чистых данных без проблем;
        - снижает оценку при наличии дефектов качества (например, дубликатов).
        """
        # Датасет без дефектов качества
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
    """Набор тестов для функции определения проблемных колонок."""

    def test_problematic_columns_found(self):
        """
        Проверка корректности идентификации проблемных колонок по пропускам.

        Тест проверяет, что функция правильно определяет колонки,
        превышающие заданный порог доли пропущенных значений.
        """
        df = pd.DataFrame({
            "good": [1, 2, 3, 4, 5],
            "bad": [None, None, None, 4, 5]
        })

        problematic = core.get_problematic_columns(df, min_missing_share=0.5)

        assert len(problematic) == 1
        assert problematic[0]["column"] == "bad"

    def test_no_problematic_columns(self):
        """
        Проверка корректности работы при отсутствии проблемных колонок.

        Тест проверяет, что функция не выявляет проблемные колонки,
        если все колонки имеют долю пропусков ниже заданного порога.
        """
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": [4, 5, 6]
        })

        problematic = core.get_problematic_columns(df, min_missing_share=0.1)

        assert len(problematic) == 0


# Расширенные тесты для дополнительных эвристик качества данных

def test_high_missing_columns_are_listed():
    """
    Тест корректности формирования списка колонок с высокой долей пропусков.

    Проверяется, что функция compute_quality_flags правильно заполняет
    поле high_missing_columns, включая в него только те колонки, доля
    пропусков в которых превышает установленный порог missing_threshold.
    """
    df = pd.DataFrame({
        "col_A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "col_B": [None, None, None, None, 5, 6, 7, 8, 9, 10],  # 40% пропусков
        "col_C": [1, 2, 3, 4, 5, 6, 7, 8, None, None]  # 20% пропусков
    })

    flags = core.compute_quality_flags(df, missing_threshold=0.3)

    assert flags["has_high_missing"] == True
    assert "col_B" in flags["high_missing_columns"]
    assert "col_C" not in flags["high_missing_columns"]


def test_zero_shares_calculation():
    """
    Тест правильности вычисления долей нулевых значений в числовых колонках.

    Проверяется корректность расчёта метрики zero_shares:
    - наличие записей для всех числовых колонок;
    - точность вычисления долей нулевых значений;
    - правильность классификации колонок как проблемных по критерию
      превышения порога zero_threshold.
    """
    df = pd.DataFrame({
        "zeros": [0, 0, 0, 0, 0, 0, 0, 0, 1, 2],  # 80% нулей
        "mixed": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],   # 10% нулей
        "no_zeros": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 0% нулей
    })

    flags = core.compute_quality_flags(df, zero_threshold=0.5)

    # Проверка наличия всех числовых колонок в словаре
    assert "zeros" in flags["zero_shares"]
    assert "mixed" in flags["zero_shares"]
    assert "no_zeros" in flags["zero_shares"]

    # Проверка точности вычислений
    assert flags["zero_shares"]["zeros"] == 0.8
    assert flags["zero_shares"]["mixed"] == 0.1
    assert flags["zero_shares"]["no_zeros"] == 0.0

    # Проверка корректности классификации проблемных колонок
    assert "zeros" in flags["high_zero_columns"]
    assert "mixed" not in flags["high_zero_columns"]
    assert "no_zeros" not in flags["high_zero_columns"]


def test_quality_score_with_all_issues():
    """
    Тест расчёта интегрального показателя качества при множественных проблемах.

    Проверяется корректность вычисления quality_score в случае одновременного
    присутствия всех типов дефектов качества данных:
    - дубликаты строк;
    - константные колонки;
    - высокая кардинальность категориальных признаков;
    - избыточная доля нулевых значений;
    - высокая доля пропущенных значений.

    Ожидается, что все соответствующие флаги будут установлены,
    а итоговая оценка качества будет существенно снижена.
    """
    df = pd.DataFrame({
        "const": ["same"] * 10,
        "high_card": [f"cat_{i}" for i in range(10)],
        "many_zeros": [0] * 7 + [1, 2, 3],
        "high_missing": [None] * 5 + [1, 2, 3, 4, 5],
        "dup_col": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    })

    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    flags = core.compute_quality_flags(
        df,
        missing_threshold=0.3,
        high_cardinality_threshold=5,
        zero_threshold=0.5
    )

    # Проверка установки всех флагов проблем
    assert flags["has_high_missing"] == True
    assert flags["has_duplicates"] == True
    assert flags["has_constant_columns"] == True
    assert flags["has_high_cardinality_categoricals"] == True
    assert flags["has_many_zero_values"] == True

    # Проверка снижения итоговой оценки
    assert flags["quality_score"] < 100
    assert flags["quality_score"] >= 40


def test_numeric_and_categorical_summary():
    """
    Тест корректности разделения признаков на числовые и категориальные.

    Проверяется правильность работы функций get_numeric_summary и
    get_categorical_summary:
    - корректное определение типов признаков;
    - правильная классификация колонок;
    - соблюдение ограничения top_k для категориальных признаков.
    """
    df = pd.DataFrame({
        "num1": [1, 2, 3, 4, 5],
        "num2": [10, 20, 30, 40, 50],
        "cat1": ["A", "B", "A", "C", "B"],
        "cat2": ["X", "Y", "Z", "X", "Y"]
    })

    numeric_summary = core.get_numeric_summary(df)
    categorical_summary = core.get_categorical_summary(df, top_k=2)

    # Проверка числовых признаков
    assert len(numeric_summary["numeric_columns"]) == 2
    assert "num1" in numeric_summary["numeric_columns"]
    assert "num2" in numeric_summary["numeric_columns"]

    # Проверка категориальных признаков
    assert len(categorical_summary["categorical_columns"]) == 2
    assert "cat1" in categorical_summary["categorical_columns"]
    assert "cat2" in categorical_summary["categorical_columns"]

    # Проверка соблюдения ограничения top_k
    assert len(categorical_summary["stats"]["cat1"]["top_values"]) <= 2


def test_problematic_columns_with_different_thresholds():
    """
    Тест работы функции определения проблемных колонок с различными порогами.

    Проверяется корректность работы функции get_problematic_columns
    при задании различных значений порога min_missing_share:
    - правильность отбора колонок в зависимости от установленного порога;
    - корректность подсчёта количества проблемных колонок;
    - адекватность поведения при изменении чувствительности критерия.
    """
    df = pd.DataFrame({
        "col1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 0% пропусков
        "col2": [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # 10% пропусков
        "col3": [None, None, None, 4, 5, 6, 7, 8, 9, 10],  # 30% пропусков
        "col4": [None, None, None, None, None, 6, 7, 8, 9, 10]  # 50% пропусков
    })

    # Тестирование с порогом 20%
    problematic_20 = core.get_problematic_columns(df, min_missing_share=0.2)
    assert len(problematic_20) == 2
    problem_cols_20 = [p["column"] for p in problematic_20]
    assert "col3" in problem_cols_20
    assert "col4" in problem_cols_20

    # Тестирование с порогом 40%
    problematic_40 = core.get_problematic_columns(df, min_missing_share=0.4)
    assert len(problematic_40) == 1
    assert problematic_40[0]["column"] == "col4"

    # Тестирование с порогом 60%
    problematic_60 = core.get_problematic_columns(df, min_missing_share=0.6)
    assert len(problematic_60) == 0