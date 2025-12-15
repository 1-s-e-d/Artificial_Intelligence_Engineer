"""CLI интерфейс для EDA утилиты."""

import sys
from pathlib import Path

# Для запуска напрямую из PyCharm - добавляем src/ в путь
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

import json
import typer
from rich.console import Console
from rich.table import Table

from eda_cli import core, viz

app = typer.Typer(help="EDA CLI - утилита для разведочного анализа данных")
console = Console()


@app.command()
def overview(
    filepath: str = typer.Argument(..., help="Путь к CSV файлу"),
    sep: str = typer.Option(",", "--sep", "-s", help="Разделитель CSV"),
    encoding: str = typer.Option("utf-8", "--encoding", "-e", help="Кодировка файла")
):
    """Быстрый обзор датасета: размеры, типы, пропуски."""
    try:
        df = core.load_csv(filepath, sep=sep, encoding=encoding)
    except FileNotFoundError as e:
        console.print(f"[red]Ошибка:[/red] {e}")
        raise typer.Exit(1)

    stats = core.get_basic_stats(df)
    missing = core.get_missing_info(df)

    console.print(f"\n[bold cyan]Обзор датасета:[/bold cyan] {filepath}\n")

    # Таблица базовой информации
    table = Table(title="Базовая статистика")
    table.add_column("Параметр", style="cyan")
    table.add_column("Значение", style="green")

    table.add_row("Строк", str(stats["n_rows"]))
    table.add_row("Колонок", str(stats["n_cols"]))
    table.add_row("Память (МБ)", str(stats["memory_mb"]))
    table.add_row("Пропусков всего", str(missing["total_missing"]))
    table.add_row("Колонок с пропусками", str(missing["columns_with_missing"]))

    console.print(table)

    # Типы данных
    console.print("\n[bold]Типы данных:[/bold]")
    for col, dtype in stats["dtypes"].items():
        console.print(f"  {col}: [yellow]{dtype}[/yellow]")


@app.command()
def report(
    filepath: str = typer.Argument(..., help="Путь к CSV файлу"),
    out_dir: str = typer.Option("reports", "--out-dir", "-o", help="Папка для отчёта"),
    sep: str = typer.Option(",", "--sep", "-s", help="Разделитель CSV"),
    encoding: str = typer.Option("utf-8", "--encoding", "-e", help="Кодировка файла"),
    max_hist_columns: int = typer.Option(6, "--max-hist-columns", help="Макс. колонок для гистограмм"),
    top_k_categories: int = typer.Option(5, "--top-k-categories", help="Top-K значений для категориальных"),
    title: str = typer.Option("EDA Report", "--title", "-t", help="Заголовок отчёта"),
    min_missing_share: float = typer.Option(0.1, "--min-missing-share", help="Порог доли пропусков для проблемных колонок"),
    json_summary: bool = typer.Option(False, "--json-summary", help="Сохранить JSON-сводку")
):
    """Генерация полного отчёта с визуализациями."""
    try:
        df = core.load_csv(filepath, sep=sep, encoding=encoding)
    except FileNotFoundError as e:
        console.print(f"[red]Ошибка:[/red] {e}")
        raise typer.Exit(1)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold cyan]Генерация отчёта...[/bold cyan]")

    # Сбор данных
    stats = core.get_basic_stats(df)
    missing = core.get_missing_info(df)
    numeric = core.get_numeric_summary(df)
    categorical = core.get_categorical_summary(df, top_k=top_k_categories)
    quality = core.compute_quality_flags(df)
    problematic = core.get_problematic_columns(df, min_missing_share=min_missing_share)

    # Визуализации
    hist_path = viz.save_histograms(df, out_path, max_columns=max_hist_columns)
    missing_path = viz.save_missing_bar(df, out_path)
    boxplot_path = viz.save_boxplots(df, out_path, max_columns=max_hist_columns)

    # Дополнительная визуализация для первой категориальной колонки
    cat_bar_path = ""
    if categorical["categorical_columns"]:
        first_cat = categorical["categorical_columns"][0]
        cat_bar_path = viz.save_category_bar(df, first_cat, out_path, top_n=top_k_categories)

    # Формирование Markdown отчёта
    report_lines = []
    report_lines.append(f"# {title}")
    report_lines.append(f"\nФайл: `{filepath}`\n")
    report_lines.append(f"Дата генерации: автоматическая\n")

    report_lines.append("## Базовая информация\n")
    report_lines.append(f"- Строк: {stats['n_rows']}")
    report_lines.append(f"- Колонок: {stats['n_cols']}")
    report_lines.append(f"- Память: {stats['memory_mb']} МБ\n")

    report_lines.append("## Качество данных\n")
    report_lines.append(f"**Quality Score:** {quality['quality_score']}/100\n")
    report_lines.append(f"- Дубликаты строк: {'Да' if quality['has_duplicates'] else 'Нет'} ({quality['duplicate_count']} шт.)")
    report_lines.append(f"- Высокая доля пропусков: {'Да' if quality['has_high_missing'] else 'Нет'}")

    # Список колонок с высокой долей пропусков
    if quality['high_missing_columns']:
        report_lines.append(f"  - Колонки с высокой долей пропусков: {', '.join(quality['high_missing_columns'])}")

    report_lines.append(f"- Константные колонки: {'Да' if quality['has_constant_columns'] else 'Нет'} ({', '.join(quality['constant_columns']) or '-'})")
    report_lines.append(f"- Высокая кардинальность: {'Да' if quality['has_high_cardinality_categoricals'] else 'Нет'} ({', '.join(quality['high_cardinality_columns']) or '-'})")
    report_lines.append(f"- Много нулей: {'Да' if quality['has_many_zero_values'] else 'Нет'} ({', '.join(quality['high_zero_columns']) or '-'})")

    # Подробная информация о долях нулей в колонках
    if quality['has_many_zero_values'] and quality['zero_shares']:
        report_lines.append("\n### Доли нулевых значений в числовых колонках\n")
        for col, share in quality['zero_shares'].items():
            if share > 0:
                report_lines.append(f"- `{col}`: {share*100:.2f}% нулей")

    report_lines.append("\n")

    report_lines.append(f"### Параметры анализа\n")
    report_lines.append(f"- Порог пропусков (min_missing_share): {min_missing_share}")
    report_lines.append(f"- Top-K категорий: {top_k_categories}")
    report_lines.append(f"- Макс. гистограмм: {max_hist_columns}\n")

    if problematic:
        report_lines.append("### Проблемные колонки (по пропускам)\n")
        for p in problematic:
            report_lines.append(f"- `{p['column']}`: {p['missing_share']*100:.1f}% пропусков ({p['missing_count']} шт.)")
        report_lines.append("")

    report_lines.append("## Пропуски\n")
    report_lines.append(f"- Всего пропусков: {missing['total_missing']}")
    report_lines.append(f"- Колонок с пропусками: {missing['columns_with_missing']}\n")

    if missing_path:
        report_lines.append(f"![Пропуски](missing_bar.png)\n")

    report_lines.append("## Числовые признаки\n")
    if numeric["numeric_columns"]:
        report_lines.append(f"Колонки: {', '.join(numeric['numeric_columns'])}\n")
        if hist_path:
            report_lines.append(f"![Гистограммы](histograms.png)\n")
        if boxplot_path:
            report_lines.append(f"![Boxplot](boxplots.png)\n")
    else:
        report_lines.append("Числовых колонок не обнаружено.\n")

    report_lines.append("## Категориальные признаки\n")
    if categorical["categorical_columns"]:
        report_lines.append(f"Колонки: {', '.join(categorical['categorical_columns'])}\n")
        for col, info in categorical["stats"].items():
            report_lines.append(f"### {col}\n")
            report_lines.append(f"- Уникальных: {info['unique_count']}")
            report_lines.append(f"- Пропусков: {info['null_count']}")
            report_lines.append(f"- Top-{top_k_categories}: {list(info['top_values'].keys())}\n")
        if cat_bar_path:
            first_cat = categorical["categorical_columns"][0]
            report_lines.append(f"![Категории {first_cat}](category_{first_cat}.png)\n")
    else:
        report_lines.append("Категориальных колонок не обнаружено.\n")

    # Сохранение отчёта
    report_file = out_path / "report.md"
    report_file.write_text("\n".join(report_lines), encoding="utf-8")

    console.print(f"[green]✓[/green] Отчёт сохранён: {report_file}")

    # JSON сводка с метаданными
    if json_summary:
        # Конвертируем все значения в нативные Python типы для JSON сериализации
        summary = {
            "n_rows": int(stats["n_rows"]),
            "n_cols": int(stats["n_cols"]),
            "quality_score": int(quality["quality_score"]),
            "total_missing": int(missing["total_missing"]),
            "problematic_columns": [p["column"] for p in problematic],
            "has_duplicates": bool(quality["has_duplicates"]),
            "has_constant_columns": bool(quality["has_constant_columns"]),
            "constant_columns": quality["constant_columns"],
            "has_high_cardinality": bool(quality["has_high_cardinality_categoricals"]),
            "high_cardinality_columns": quality["high_cardinality_columns"],
            "high_missing_columns": quality["high_missing_columns"],
            "zero_shares": {k: float(v) for k, v in quality["zero_shares"].items()}
        }

        json_file = out_path / "summary.json"
        json_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        console.print(f"[green]✓[/green] JSON-сводка: {json_file}")


@app.command()
def head(
    filepath: str = typer.Argument(..., help="Путь к CSV файлу"),
    n: int = typer.Option(5, "--n", "-n", help="Количество строк"),
    sep: str = typer.Option(",", "--sep", "-s", help="Разделитель CSV"),
    encoding: str = typer.Option("utf-8", "--encoding", "-e", help="Кодировка файла")
):
    """Вывод первых N строк датасета."""
    try:
        df = core.load_csv(filepath, sep=sep, encoding=encoding)
    except FileNotFoundError as e:
        console.print(f"[red]Ошибка:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]Первые {n} строк:[/bold cyan] {filepath}\n")

    table = Table()
    for col in df.columns:
        table.add_column(col, style="cyan", overflow="fold")

    for idx, row in df.head(n).iterrows():
        table.add_row(*[str(v) for v in row.values])

    console.print(table)


@app.command()
def sample(
    filepath: str = typer.Argument(..., help="Путь к CSV файлу"),
    n: int = typer.Option(5, "--n", "-n", help="Количество строк"),
    sep: str = typer.Option(",", "--sep", "-s", help="Разделитель CSV"),
    encoding: str = typer.Option("utf-8", "--encoding", "-e", help="Кодировка файла"),
    seed: int = typer.Option(None, "--seed", help="Seed для воспроизводимости")
):
    """Вывод случайной выборки N строк."""
    try:
        df = core.load_csv(filepath, sep=sep, encoding=encoding)
    except FileNotFoundError as e:
        console.print(f"[red]Ошибка:[/red] {e}")
        raise typer.Exit(1)

    n_actual = min(n, len(df))
    sample_df = df.sample(n=n_actual, random_state=seed)

    console.print(f"\n[bold cyan]Случайная выборка ({n_actual} строк):[/bold cyan] {filepath}\n")

    table = Table()
    for col in df.columns:
        table.add_column(col, style="cyan", overflow="fold")

    for idx, row in sample_df.iterrows():
        table.add_row(*[str(v) for v in row.values])

    console.print(table)


def run_multiple_reports():
    """Функция для генерации нескольких отчётов с разными параметрами."""
    project_root = Path(__file__).parent.parent.parent
    data_file = str(project_root / "data" / "example.csv")

    console.print("[bold green]═══════════════════════════════════════════════════════════[/bold green]")
    console.print("[bold green]    Генерация отчётов EDA (запуск из PyCharm)             [/bold green]")
    console.print("[bold green]═══════════════════════════════════════════════════════════[/bold green]\n")

    # Отчёт 1: Базовый (reports_example)
    console.print("[bold cyan] Отчёт 1: Базовый вариант[/bold cyan]")
    try:
        df = core.load_csv(data_file)
        stats = core.get_basic_stats(df)
        missing = core.get_missing_info(df)
        numeric = core.get_numeric_summary(df)
        categorical = core.get_categorical_summary(df, top_k=5)
        quality = core.compute_quality_flags(df)
        problematic = core.get_problematic_columns(df, min_missing_share=0.1)

        out_path = project_root / "reports_example"
        out_path.mkdir(parents=True, exist_ok=True)

        viz.save_histograms(df, out_path, max_columns=6)
        viz.save_missing_bar(df, out_path)
        viz.save_boxplots(df, out_path, max_columns=6)

        if categorical["categorical_columns"]:
            first_cat = categorical["categorical_columns"][0]
            viz.save_category_bar(df, first_cat, out_path, top_n=5)

        report_lines = []
        report_lines.append("# EDA Report (базовый)")
        report_lines.append(f"\nФайл: `{data_file}`\n")
        report_lines.append("## Базовая информация\n")
        report_lines.append(f"- Строк: {stats['n_rows']}")
        report_lines.append(f"- Колонок: {stats['n_cols']}")
        report_lines.append(f"- Память: {stats['memory_mb']} МБ\n")
        report_lines.append("## Качество данных\n")
        report_lines.append(f"**Quality Score:** {quality['quality_score']}/100\n")
        report_lines.append(f"- Дубликаты строк: {'Да' if quality['has_duplicates'] else 'Нет'} ({quality['duplicate_count']} шт.)")
        report_lines.append(f"- Высокая доля пропусков: {'Да' if quality['has_high_missing'] else 'Нет'}")
        if quality['high_missing_columns']:
            report_lines.append(f"  - Колонки с высокой долей пропусков: {', '.join(quality['high_missing_columns'])}")
        report_lines.append(f"- Константные колонки: {'Да' if quality['has_constant_columns'] else 'Нет'} ({', '.join(quality['constant_columns']) or '-'})")
        report_lines.append(f"- Высокая кардинальность: {'Да' if quality['has_high_cardinality_categoricals'] else 'Нет'} ({', '.join(quality['high_cardinality_columns']) or '-'})")
        report_lines.append(f"- Много нулей: {'Да' if quality['has_many_zero_values'] else 'Нет'} ({', '.join(quality['high_zero_columns']) or '-'})")
        if quality['has_many_zero_values'] and quality['zero_shares']:
            report_lines.append("\n### Доли нулевых значений в числовых колонках\n")
            for col, share in quality['zero_shares'].items():
                if share > 0:
                    report_lines.append(f"- `{col}`: {share*100:.2f}% нулей")
        report_lines.append("\n")
        if problematic:
            report_lines.append("### Проблемные колонки (по пропускам)\n")
            for p in problematic:
                report_lines.append(f"- `{p['column']}`: {p['missing_share']*100:.1f}% пропусков ({p['missing_count']} шт.)")
            report_lines.append("")
        report_lines.append("## Пропуски\n")
        report_lines.append(f"- Всего пропусков: {missing['total_missing']}")
        report_lines.append(f"- Колонок с пропусками: {missing['columns_with_missing']}\n")
        report_lines.append(f"![Пропуски](missing_bar.png)\n")
        report_lines.append("## Числовые признаки\n")
        if numeric["numeric_columns"]:
            report_lines.append(f"Колонки: {', '.join(numeric['numeric_columns'])}\n")
            report_lines.append(f"![Гистограммы](histograms.png)\n")
            report_lines.append(f"![Boxplot](boxplots.png)\n")
        report_lines.append("## Категориальные признаки\n")
        if categorical["categorical_columns"]:
            report_lines.append(f"Колонки: {', '.join(categorical['categorical_columns'])}\n")
            for col, info in categorical["stats"].items():
                report_lines.append(f"### {col}\n")
                report_lines.append(f"- Уникальных: {info['unique_count']}")
                report_lines.append(f"- Пропусков: {info['null_count']}")
                report_lines.append(f"- Top-5: {list(info['top_values'].keys())}\n")
            if categorical["categorical_columns"]:
                first_cat = categorical["categorical_columns"][0]
                report_lines.append(f"![Категории {first_cat}](category_{first_cat}.png)\n")

        report_file = out_path / "report.md"
        report_file.write_text("\n".join(report_lines), encoding="utf-8")
        console.print(f"[green]✓[/green] Базовый отчёт: {out_path}/report.md\n")
    except Exception as e:
        console.print(f"[red]✗[/red] Ошибка генерации базового отчёта: {e}\n")

    # Отчёт 2: Расширенный (reports_custom)
    console.print("[bold cyan] Отчёт 2: Расширенный вариант с JSON[/bold cyan]")
    try:
        df = core.load_csv(data_file)
        stats = core.get_basic_stats(df)
        missing = core.get_missing_info(df)
        numeric = core.get_numeric_summary(df)
        categorical = core.get_categorical_summary(df, top_k=3)
        quality = core.compute_quality_flags(df)
        problematic = core.get_problematic_columns(df, min_missing_share=0.05)

        out_path = project_root / "reports_custom"
        out_path.mkdir(parents=True, exist_ok=True)

        viz.save_histograms(df, out_path, max_columns=4)
        viz.save_missing_bar(df, out_path)
        viz.save_boxplots(df, out_path, max_columns=4)

        if categorical["categorical_columns"]:
            first_cat = categorical["categorical_columns"][0]
            viz.save_category_bar(df, first_cat, out_path, top_n=3)

        report_lines = []
        report_lines.append("# HW03: анализ example.csv")
        report_lines.append(f"\nФайл: `{data_file}`\n")
        report_lines.append("## Базовая информация\n")
        report_lines.append(f"- Строк: {stats['n_rows']}")
        report_lines.append(f"- Колонок: {stats['n_cols']}")
        report_lines.append(f"- Память: {stats['memory_mb']} МБ\n")
        report_lines.append("## Качество данных\n")
        report_lines.append(f"**Quality Score:** {quality['quality_score']}/100\n")
        report_lines.append(f"- Дубликаты строк: {'Да' if quality['has_duplicates'] else 'Нет'} ({quality['duplicate_count']} шт.)")
        report_lines.append(f"- Высокая доля пропусков: {'Да' if quality['has_high_missing'] else 'Нет'}")
        if quality['high_missing_columns']:
            report_lines.append(f"  - Колонки с высокой долей пропусков: {', '.join(quality['high_missing_columns'])}")
        report_lines.append(f"- Константные колонки: {'Да' if quality['has_constant_columns'] else 'Нет'} ({', '.join(quality['constant_columns']) or '-'})")
        report_lines.append(f"- Высокая кардинальность: {'Да' if quality['has_high_cardinality_categoricals'] else 'Нет'} ({', '.join(quality['high_cardinality_columns']) or '-'})")
        report_lines.append(f"- Много нулей: {'Да' if quality['has_many_zero_values'] else 'Нет'} ({', '.join(quality['high_zero_columns']) or '-'})")
        if quality['has_many_zero_values'] and quality['zero_shares']:
            report_lines.append("\n### Доли нулевых значений в числовых колонках\n")
            for col, share in quality['zero_shares'].items():
                if share > 0:
                    report_lines.append(f"- `{col}`: {share*100:.2f}% нулей")
        report_lines.append("\n")
        report_lines.append(f"### Параметры анализа\n")
        report_lines.append(f"- Порог пропусков (min_missing_share): 0.05")
        report_lines.append(f"- Top-K категорий: 3")
        report_lines.append(f"- Макс. гистограмм: 4\n")
        if problematic:
            report_lines.append("### Проблемные колонки (по пропускам)\n")
            for p in problematic:
                report_lines.append(f"- `{p['column']}`: {p['missing_share']*100:.1f}% пропусков ({p['missing_count']} шт.)")
            report_lines.append("")
        report_lines.append("## Пропуски\n")
        report_lines.append(f"- Всего пропусков: {missing['total_missing']}")
        report_lines.append(f"- Колонок с пропусками: {missing['columns_with_missing']}\n")
        report_lines.append(f"![Пропуски](missing_bar.png)\n")
        report_lines.append("## Числовые признаки\n")
        if numeric["numeric_columns"]:
            report_lines.append(f"Колонки: {', '.join(numeric['numeric_columns'])}\n")
            report_lines.append(f"![Гистограммы](histograms.png)\n")
            report_lines.append(f"![Boxplot](boxplots.png)\n")
        report_lines.append("## Категориальные признаки\n")
        if categorical["categorical_columns"]:
            report_lines.append(f"Колонки: {', '.join(categorical['categorical_columns'])}\n")
            for col, info in categorical["stats"].items():
                report_lines.append(f"### {col}\n")
                report_lines.append(f"- Уникальных: {info['unique_count']}")
                report_lines.append(f"- Пропусков: {info['null_count']}")
                report_lines.append(f"- Top-3: {list(info['top_values'].keys())}\n")
            if categorical["categorical_columns"]:
                first_cat = categorical["categorical_columns"][0]
                report_lines.append(f"![Категории {first_cat}](category_{first_cat}.png)\n")

        report_file = out_path / "report.md"
        report_file.write_text("\n".join(report_lines), encoding="utf-8")

        # JSON-сводка
        summary = {
            "n_rows": int(stats["n_rows"]),
            "n_cols": int(stats["n_cols"]),
            "quality_score": int(quality["quality_score"]),
            "total_missing": int(missing["total_missing"]),
            "problematic_columns": [p["column"] for p in problematic],
            "has_duplicates": bool(quality["has_duplicates"]),
            "has_constant_columns": bool(quality["has_constant_columns"]),
            "constant_columns": quality["constant_columns"],
            "has_high_cardinality": bool(quality["has_high_cardinality_categoricals"]),
            "high_cardinality_columns": quality["high_cardinality_columns"],
            "high_missing_columns": quality["high_missing_columns"],
            "zero_shares": {k: float(v) for k, v in quality["zero_shares"].items()}
        }

        json_file = out_path / "summary.json"
        json_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

        console.print(f"[green]✓[/green] Расширенный отчёт: {out_path}/report.md")
        console.print(f"[green]✓[/green] JSON-сводка: {out_path}/summary.json\n")
    except Exception as e:
        console.print(f"[red]✗[/red] Ошибка генерации расширенного отчёта: {e}\n")

    console.print("[bold green]═══════════════════════════════════════════════════════════[/bold green]")
    console.print("[bold green]    ✓ Генерация завершена успешно!                         [/bold green]")
    console.print("[bold green]═══════════════════════════════════════════════════════════[/bold green]\n")


if __name__ == "__main__":
    # Автоматический запуск при нажатии кнопки Run в PyCharm
    if len(sys.argv) == 1:
        # Запуск генерации двух отчётов
        run_multiple_reports()
    else:
        # Запуск через CLI с аргументами
        app()
