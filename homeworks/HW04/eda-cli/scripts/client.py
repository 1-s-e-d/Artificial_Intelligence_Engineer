"""
Клиент для тестирования API качества датасетов.
Вариант E из HW04.
"""
import json
from pathlib import Path
import requests
from rich.console import Console
from rich.table import Table

console = Console()

# URL сервиса
BASE_URL = "http://127.0.0.1:8001"


def test_health():
    """Проверка health-check эндпоинта."""
    console.print("\n[bold cyan]1. Проверка /health[/bold cyan]")

    try:
        response = requests.get(f"{BASE_URL}/health")
        console.print(f"Статус: [green]{response.status_code}[/green]")
        console.print(f"Ответ: {response.json()}")
        return True
    except Exception as e:
        console.print(f"[red]Ошибка:[/red] {e}")
        return False


def test_quality():
    """Тестирование /quality с разными параметрами."""
    console.print("\n[bold cyan]2. Тестирование /quality[/bold cyan]")

    test_cases = [
        {
            "name": "Хороший датасет",
            "data": {
                "n_rows": 5000,
                "n_cols": 10,
                "max_missing_share": 0.1,
                "numeric_cols": 6,
                "categorical_cols": 4
            }
        },
        {
            "name": "Маленький датасет",
            "data": {
                "n_rows": 500,
                "n_cols": 5,
                "max_missing_share": 0.05,
                "numeric_cols": 3,
                "categorical_cols": 2
            }
        },
        {
            "name": "Много пропусков",
            "data": {
                "n_rows": 2000,
                "n_cols": 15,
                "max_missing_share": 0.6,
                "numeric_cols": 8,
                "categorical_cols": 7
            }
        },
    ]

    results = []

    for i, test_case in enumerate(test_cases, 1):
        console.print(f"\n  Тест {i}: {test_case['name']}")

        try:
            response = requests.post(
                f"{BASE_URL}/quality",
                json=test_case["data"]
            )

            if response.status_code == 200:
                result = response.json()
                results.append({
                    "name": test_case["name"],
                    "status": response.status_code,
                    "ok_for_model": result["ok_for_model"],
                    "quality_score": result["quality_score"],
                    "latency_ms": result["latency_ms"]
                })

                console.print(f"    Статус: [green]{response.status_code}[/green]")
                console.print(
                    f"    OK для модели: [{'green' if result['ok_for_model'] else 'red'}]{result['ok_for_model']}[/{'green' if result['ok_for_model'] else 'red'}]")
                console.print(f"    Quality Score: {result['quality_score']:.2f}")
                console.print(f"    Latency: {result['latency_ms']:.2f} ms")
            else:
                console.print(f"    [red]Ошибка {response.status_code}[/red]")
                results.append({
                    "name": test_case["name"],
                    "status": response.status_code,
                    "ok_for_model": None,
                    "quality_score": None,
                    "latency_ms": None
                })

        except Exception as e:
            console.print(f"    [red]Исключение:[/red] {e}")
            results.append({
                "name": test_case["name"],
                "status": "ERROR",
                "ok_for_model": None,
                "quality_score": None,
                "latency_ms": None
            })

    return results


def test_quality_from_csv():
    """Тестирование /quality-from-csv."""
    console.print("\n[bold cyan]3. Тестирование /quality-from-csv[/bold cyan]")

    # Путь к тестовому файлу
    csv_path = Path("data/example.csv")

    if not csv_path.exists():
        console.print(f"[red]Файл {csv_path} не найден[/red]")
        return None

    try:
        with open(csv_path, "rb") as f:
            files = {"file": ("example.csv", f, "text/csv")}
            response = requests.post(
                f"{BASE_URL}/quality-from-csv",
                files=files
            )

        if response.status_code == 200:
            result = response.json()
            console.print(f"Статус: [green]{response.status_code}[/green]")
            console.print(
                f"OK для модели: [{'green' if result['ok_for_model'] else 'red'}]{result['ok_for_model']}[/{'green' if result['ok_for_model'] else 'red'}]")
            console.print(f"Quality Score: {result['quality_score']:.2f}")
            console.print(
                f"Размер датасета: {result['dataset_shape']['n_rows']} строк × {result['dataset_shape']['n_cols']} колонок")
            console.print(f"Latency: {result['latency_ms']:.2f} ms")
            console.print(f"Флаги: {result['flags']}")
            return result
        else:
            console.print(f"[red]Ошибка {response.status_code}[/red]: {response.text}")
            return None

    except Exception as e:
        console.print(f"[red]Исключение:[/red] {e}")
        return None


def test_quality_flags_from_csv():
    """Тестирование /quality-flags-from-csv (новый эндпоинт HW04)."""
    console.print("\n[bold cyan]4. Тестирование /quality-flags-from-csv (HW04)[/bold cyan]")

    csv_path = Path("data/example.csv")

    if not csv_path.exists():
        console.print(f"[red]Файл {csv_path} не найден[/red]")
        return None

    try:
        with open(csv_path, "rb") as f:
            files = {"file": ("example.csv", f, "text/csv")}
            response = requests.post(
                f"{BASE_URL}/quality-flags-from-csv",
                files=files
            )

        if response.status_code == 200:
            result = response.json()
            console.print(f"Статус: [green]{response.status_code}[/green]")
            console.print(f"Quality Score: {result['quality_score']}/100")
            console.print(
                f"Размер датасета: {result['dataset_shape']['n_rows']} строк × {result['dataset_shape']['n_cols']} колонок")
            console.print(f"Latency: {result['latency_ms']:.2f} ms")
            console.print(f"\nВсе флаги качества:")
            console.print(json.dumps(result['flags'], indent=2, ensure_ascii=False))
            return result
        else:
            console.print(f"[red]Ошибка {response.status_code}[/red]: {response.text}")
            return None

    except Exception as e:
        console.print(f"[red]Исключение:[/red] {e}")
        return None


def test_metrics():
    """Проверка /metrics эндпоинта."""
    console.print("\n[bold cyan]5. Проверка /metrics[/bold cyan]")

    try:
        response = requests.get(f"{BASE_URL}/metrics")
        console.print(f"Статус: [green]{response.status_code}[/green]")

        if response.status_code == 200:
            metrics = response.json()
            console.print(f"\nМетрики сервиса:")
            console.print(f"  Всего запросов: {metrics['total_requests']}")
            console.print(f"  Средняя задержка: {metrics['avg_latency_ms']:.2f} ms")
            console.print(f"  Количество ошибок: {metrics['errors']}")
            console.print(f"  Последний ok_for_model: {metrics['last_ok_for_model']}")
            console.print(f"  Вызовы по эндпоинтам: {metrics['endpoint_calls']}")
            return metrics
        else:
            console.print(f"[red]Ошибка {response.status_code}[/red]")
            return None

    except Exception as e:
        console.print(f"[red]Ошибка:[/red] {e}")
        return None


def print_summary(quality_results):
    """Вывод сводной таблицы результатов."""
    console.print("\n[bold green]══════════════════════════════════════════════════[/bold green]")
    console.print("[bold green]           Сводка по тестам /quality            [/bold green]")
    console.print("[bold green]══════════════════════════════════════════════════[/bold green]\n")

    table = Table(title="Результаты тестирования")

    table.add_column("Тест", style="cyan")
    table.add_column("Статус", style="green")
    table.add_column("OK для модели", justify="center")
    table.add_column("Quality Score", justify="right")
    table.add_column("Latency (ms)", justify="right")

    for result in quality_results:
        ok_style = "green" if result["ok_for_model"] else "red"
        ok_text = "✓" if result["ok_for_model"] else "✗" if result["ok_for_model"] is not None else "-"

        table.add_row(
            result["name"],
            str(result["status"]),
            f"[{ok_style}]{ok_text}[/{ok_style}]",
            f"{result['quality_score']:.2f}" if result["quality_score"] is not None else "-",
            f"{result['latency_ms']:.2f}" if result["latency_ms"] is not None else "-"
        )

    console.print(table)


def main():
    """Главная функция - запуск всех тестов."""
    console.print("[bold magenta]═══════════════════════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]   Клиент для тестирования API качества датасетов    [/bold magenta]")
    console.print("[bold magenta]═══════════════════════════════════════════════════════[/bold magenta]")

    # Проверяем доступность сервиса
    if not test_health():
        console.print("\n[red]Сервис недоступен! Убедитесь, что сервер запущен на {BASE_URL}[/red]")
        return

    # Запускаем тесты
    quality_results = test_quality()
    test_quality_from_csv()
    test_quality_flags_from_csv()
    test_metrics()

    # Выводим сводку
    print_summary(quality_results)

    console.print("\n[bold green]✓ Все тесты завершены![/bold green]\n")


if __name__ == "__main__":
    main()