"""CLI интерфейс для EDA утилиты."""

import json
import typer
from pathlib import Path
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
        # Новые параметры CLI
        max_hist_columns: int = typer.Option(6, "--max-hist-columns", help="Макс. колонок для гистограмм"),
        top_k_categories: int = typer.Option(5, "--top-k-categories", help="Top-K значений для категориальных"),
        title: str = typer.Option("EDA Report", "--title", "-t", help="Заголовок отчёта"),
        min_missing_share: float = typer.Option(0.1, "--min-missing-share",
                                                help="Порог доли пропусков для проблемных колонок"),
        json_summary: bool = typer.Option(False, "--json-summary", help="Сохранить JSON-сводку (Вариант B)")
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

    # Доп. визуализация для первой категориальной колонки (Вариант C)
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
    report_lines.append(
        f"- Дубликаты строк: {'Да' if quality['has_duplicates'] else 'Нет'} ({quality['duplicate_count']} шт.)")
    report_lines.append(f"- Высокая доля пропусков: {'Да' if quality['has_high_missing'] else 'Нет'}")
    report_lines.append(
        f"- Константные колонки: {'Да' if quality['has_constant_columns'] else 'Нет'} ({', '.join(quality['constant_columns']) or '-'})")
    report_lines.append(
        f"- Высокая кардинальность: {'Да' if quality['has_high_cardinality_categoricals'] else 'Нет'} ({', '.join(quality['high_cardinality_columns']) or '-'})")
    report_lines.append(
        f"- Много нулей: {'Да' if quality['has_many_zero_values'] else 'Нет'} ({', '.join(quality['high_zero_columns']) or '-'})\n")

    report_lines.append(f"### Параметры анализа\n")
    report_lines.append(f"- Порог пропусков (min_missing_share): {min_missing_share}")
    report_lines.append(f"- Top-K категорий: {top_k_categories}")
    report_lines.append(f"- Макс. гистограмм: {max_hist_columns}\n")

    if problematic:
        report_lines.append("### Проблемные колонки (по пропускам)\n")
        for p in problematic:
            report_lines.append(
                f"- `{p['column']}`: {p['missing_share'] * 100:.1f}% пропусков ({p['missing_count']} шт.)")
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

    # Сохранение report.md
    report_file = out_path / "report.md"
    report_file.write_text("\n".join(report_lines), encoding="utf-8")

    console.print(f"[green]✓[/green] Отчёт сохранён: {report_file}")

    # JSON сводка (Вариант B)
    if json_summary:
        summary = {
            "n_rows": stats["n_rows"],
            "n_cols": stats["n_cols"],
            "quality_score": quality["quality_score"],
            "total_missing": missing["total_missing"],
            "problematic_columns": [p["column"] for p in problematic],
            "has_duplicates": quality["has_duplicates"],
            "has_constant_columns": quality["has_constant_columns"],
            "constant_columns": quality["constant_columns"],
            "has_high_cardinality": quality["has_high_cardinality_categoricals"],
            "high_cardinality_columns": quality["high_cardinality_columns"]
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
    """Вывод первых N строк датасета (Вариант A)."""
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
    """Вывод случайной выборки N строк (Вариант A)."""
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


if __name__ == "__main__":
    app()
