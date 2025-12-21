# HW04 — HTTP API для оценки качества датасетов

Домашнее задание к семинару S04: расширение EDA CLI с реализацией HTTP API на `FastAPI` для предоставления функций оценки качества данных через веб-интерфейс. Добавлены структурированное логирование, метрики и интеграция всех компонентов.

---

## 1. Цель работы

- Расширить EDA CLI проекта из HW03 с добавлением HTTP API сервиса.
- Реализовать несколько вариантов решения (D, E, F):
  - **Вариант D** — структурированное логирование запросов в JSON-формате.
  - **Вариант E** — клиентский скрипт для тестирования API (опционально).
  - **Вариант F** — эндпоинт `/metrics` для сбора статистики работы сервиса.
- Обеспечить интеграцию EDA-ядра из HW03 с FastAPI.
- Провести полное тестирование всех компонентов.

---

## 2. Окружение

- Python 3.12
- `uv` как менеджер проекта и зависимостей
- FastAPI (≥0.104.0)
- Uvicorn (ASGI сервер)
- ОС: Windows (запуск в PowerShell / PyCharm)

Перед началом работы нужно, чтобы команды:

```bash
uv --version
python --version
```

выдавали результаты без ошибок.

---

## 3. Структура проекта

Проект развивает структуру из HW03 с добавлением API-компонента:

```text
homeworks/
  HW04/
    eda-cli/
      pyproject.toml
      uv.lock
      README.md
      .gitignore
      src/
        eda_cli/
          __init__.py
          core.py           # EDA-ядро из HW03
          viz.py            # Визуализация из HW03
          cli.py            # CLI команды из HW03
          api.py            # ← НОВОЕ: FastAPI приложение (HW04)
      tests/
        test_core.py        # Тесты из HW03
      scripts/
        client.py           # ← НОВОЕ: клиент для тестирования (Вариант E)
      data/
        example.csv
      logs/                 # ← НОВОЕ: структурированные логи (Вариант D)
        api.log
      reports_example/
      reports_final/
      .venv/                # локальное окружение (не коммитится)
```

---

## 4. Установка зависимостей

Из корня проекта `eda-cli`:

```bash
cd homeworks/HW04/eda-cli
uv sync
```

Команда создаст виртуальное окружение `.venv` и установит все зависимости, зафиксированные в `pyproject.toml` и `uv.lock`.

---

## 5. Запуск HTTP API сервера

Сервер работает на порту `8001` по адресу `http://127.0.0.1:8001`.

### 5.1. Запуск в режиме разработки (с автоперезагрузкой)

```bash
uv run uvicorn eda_cli.api:app --reload --port 8001
```

**Ожидаемый вывод:**

```
INFO:     Uvicorn running on http://127.0.0.1:8001 (Press CTRL+C to quit)
INFO:     Started reloader process [xxxxx] using WatchFiles
INFO:     Started server process [xxxxx]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 5.2. Интерактивная документация (Swagger UI)

После запуска сервера откройте в браузере:

```
http://127.0.0.1:8001/docs
```

Здесь вы увидите все эндпоинты API, сможете тестировать их прямо из браузера, просматривать примеры запросов и ответов.

---

## 6. API Эндпоинты

### 6.1. GET /health

**Назначение:** Health-check сервиса.

**Параметры:** отсутствуют.

**Ответ (200 OK):**

```json
{
  "status": "ok",
  "service": "dataset-quality",
  "version": "0.3.0"
}
```

**Пример cURL:**

```bash
curl -X GET "http://127.0.0.1:8001/health"
```

---

### 6.2. POST /quality

**Назначение:** Эвристическая оценка качества датасета по агрегированным числовым признакам (заглушка).

**Параметры (Request Body, JSON):**

```json
{
  "n_rows": 2000,
  "n_cols": 15,
  "max_missing_share": 0.2,
  "numeric_cols": 8,
  "categorical_cols": 7
}
```

| Поле | Тип | Описание |
|------|-----|---------|
| `n_rows` | int | Количество строк в датасете (≥0) |
| `n_cols` | int | Количество колонок (≥0) |
| `max_missing_share` | float | Максимальная доля пропусков среди колонок (0.0–1.0) |
| `numeric_cols` | int | Количество числовых колонок (≥0) |
| `categorical_cols` | int | Количество категориальных колонок (≥0) |

**Ответ (200 OK):**

```json
{
  "ok_for_model": true,
  "quality_score": 0.8,
  "message": "Данных достаточно, модель можно обучать (по текущим эвристикам).",
  "latency_ms": 0.5,
  "flags": {
    "too_few_rows": false,
    "too_many_columns": false,
    "too_many_missing": false,
    "no_numeric_columns": false,
    "no_categorical_columns": false
  },
  "dataset_shape": {
    "n_rows": 2000,
    "n_cols": 15
  }
}
```

**Логирование:** Каждый запрос логируется в `logs/api.log` (Вариант D).

---

### 6.3. POST /quality-from-csv

**Назначение:** Реальная оценка качества датасета через загрузку CSV-файла с использованием EDA-ядра из HW03.

**Параметры (Request Body, multipart/form-data):**

- `file` (required) — CSV-файл для анализа

**Ответ (200 OK):**

```json
{
  "ok_for_model": true,
  "quality_score": 1.0,
  "message": "CSV выглядит достаточно качественным для обучения модели (по текущим эвристикам).",
  "latency_ms": 42.95,
  "flags": {
    "has_high_missing": false,
    "has_duplicates": false,
    "has_constant_columns": false,
    "has_high_cardinality_categoricals": false,
    "has_many_zero_values": false
  },
  "dataset_shape": {
    "n_rows": 15,
    "n_cols": 7
  }
}
```

**Ошибки:**

- `400` — неверный формат файла или невозможно прочитать CSV;
- `400` — CSV-файл пуст.

**Пример cURL:**

```bash
curl -X POST "http://127.0.0.1:8001/quality-from-csv" \
  -H "accept: application/json" \
  -F "file=@data/example.csv"
```

---

### 6.4. POST /quality-flags-from-csv (HW04, основной эндпоинт)

**Назначение:** Получить полный набор флагов качества из CSV-файла. Возвращает все эвристики из HW03 с подробной информацией о проблемных колонках.

**Параметры (Request Body, multipart/form-data):**

- `file` (required) — CSV-файл для анализа

**Ответ (200 OK):**

```json
{
  "flags": {
    "has_high_missing": false,
    "high_missing_columns": [],
    "has_duplicates": false,
    "duplicate_count": 0,
    "has_constant_columns": false,
    "constant_columns": [],
    "has_high_cardinality_categoricals": false,
    "high_cardinality_columns": [],
    "has_many_zero_values": false,
    "high_zero_columns": [],
    "zero_shares": {
      "id": 0,
      "age": 0,
      "salary": 0.2,
      "score": 0
    }
  },
  "quality_score": 100,
  "latency_ms": 4.25,
  "dataset_shape": {
    "n_rows": 15,
    "n_cols": 7
  }
}
```

**Описание полей `flags`:**

- `has_high_missing` (bool) — есть ли колонки с высокой долей пропусков;
- `high_missing_columns` (list) — список таких колонок;
- `has_duplicates` (bool) — есть ли дублирующиеся строки;
- `duplicate_count` (int) — количество дубликатов;
- `has_constant_columns` (bool) — есть ли константные колонки (одно значение);
- `constant_columns` (list) — список константных колонок;
- `has_high_cardinality_categoricals` (bool) — категориальные колонки с высокой кардинальностью;
- `high_cardinality_columns` (list) — список таких колонок;
- `has_many_zero_values` (bool) — числовые колонки с большим количеством нулей;
- `high_zero_columns` (list) — список таких колонок;
- `zero_shares` (dict) — доля нулей по каждой числовой колонке.

**Пример cURL:**

```bash
curl -X POST "http://127.0.0.1:8001/quality-flags-from-csv" \
  -H "accept: application/json" \
  -F "file=@data/example.csv"
```

---

### 6.5. GET /metrics (Вариант F — статистика сервиса)

**Назначение:** Получить метрики работы сервиса (количество запросов, среднюю задержку, количество ошибок и т.д.).

**Параметры:** отсутствуют.

**Ответ (200 OK):**

```json
{
  "total_requests": 4,
  "avg_latency_ms": 12.44,
  "endpoint_calls": {
    "quality": 1,
    "quality-from-csv": 1,
    "quality-flags-from-csv": 1,
    "health": 1
  },
  "last_ok_for_model": true,
  "errors": 0
}
```

| Поле | Описание |
|------|----------|
| `total_requests` | Общее количество обработанных запросов (всех типов) |
| `avg_latency_ms` | Средняя задержка обработки запроса в миллисекундах |
| `endpoint_calls` | Словарь с количеством вызовов каждого эндпоинта |
| `last_ok_for_model` | Последнее значение флага `ok_for_model` |
| `errors` | Количество ошибок (статус ≠ 200) |

**Пример cURL:**

```bash
curl -X GET "http://127.0.0.1:8001/metrics"
```

---

## 7. Структурированное логирование (Вариант D)

Все запросы к API логируются в файл `logs/api.log` в JSON-формате. Это позволяет легко парсить и анализировать историю запросов.

### Структура лога

Каждая строка в `logs/api.log` — это JSON объект:

```json
{
  "timestamp": "2025-12-21T17:06:35.341133",
  "request_id": "149602da-dc90-4bdb-bd7d-b36061036a8f",
  "endpoint": "quality",
  "status": 200,
  "latency_ms": 0.01,
  "n_rows": 2000,
  "n_cols": 15,
  "ok_for_model": true,
  "quality_score": 0.8
}
```

| Поле | Описание |
|------|----------|
| `timestamp` | ISO 8601 время запроса |
| `request_id` | Уникальный UUID для каждого запроса |
| `endpoint` | Название эндпоинта (quality, quality-from-csv и т.д.) |
| `status` | HTTP статус ответа (200, 400 и т.д.) |
| `latency_ms` | Время обработки запроса в миллисекундах |
| Дополнительные поля | В зависимости от типа запроса (n_rows, n_cols, filename и т.д.) |

### Просмотр логов

```bash
# Просмотр всех логов
cat logs/api.log

# Поиск ошибок
grep '"status": 40' logs/api.log

# Красивый вывод (если установлен jq)
cat logs/api.log | jq .
```

---

## 8. CLI-интерфейс (из HW03)

Проект сохраняет все CLI команды из HW03:

### 8.1. Команда `overview`

```bash
uv run eda-cli overview data/example.csv
```

Вывод: базовая статистика (размер, типы данных, пропуски).

### 8.2. Команда `report`

```bash
uv run eda-cli report data/example.csv --out-dir reports_final
```

Генерирует полный EDA-отчёт в Markdown и PNG-графики.

### 8.3. Команда `head`

```bash
uv run eda-cli head data/example.csv --n 5
```

Выводит первые 5 строк датасета в табличном виде.

### 8.4. Команда `sample`

```bash
uv run eda-cli sample data/example.csv --n 3
```

Выводит случайную выборку из 3 строк.

---

## 9. Клиентский скрипт (Вариант E)

**Опционально:** в папке `scripts/` расположен Python-скрипт `client.py`, который автоматически тестирует все эндпоинты API и выводит красивую сводку результатов через библиотеку `rich`.

### Запуск клиента

Убедитесь, что сервер запущен на `http://127.0.0.1:8001`, затем в новом терминале:

```bash
cd homeworks/HW04/eda-cli
uv run python scripts/client.py
```

**Клиент выполняет:**

1. Проверку health-check эндпоинта.
2. Тестирование `/quality` с различными параметрами.
3. Тестирование `/quality-from-csv` с загрузкой файла.
4. Тестирование `/quality-flags-from-csv` (новый HW04 эндпоинт).
5. Проверку `/metrics`.
6. Выводит сводную таблицу результатов.

---

## 10. Тесты

Все тесты из HW03 продолжают работать без изменений. Модульные тесты лежат в `tests/test_core.py`.

### Запуск тестов

```bash
cd homeworks/HW04/eda-cli
uv run pytest -v
```

или краткий вариант:

```bash
uv run pytest -q
```

**Ожидаемый результат:**

```
19 passed in 0.34s
```

Все тесты покрывают функции из `core.py` (EDA-ядро):
- загрузку CSV,
- базовую статистику,
- информацию о пропусках,
- эвристики качества (дубликаты, константные колонки, кардинальность, нули),
- расчёт `quality_score`.

---

## 11. Зависимости

Проект использует:

**Основные:**

- **pandas** (≥2.0.0) — работа с табличными данными;
- **matplotlib** (≥3.7.0) — построение графиков;
- **typer** (≥0.9.0) — создание CLI;
- **rich** (≥13.0.0) — красивый вывод в консоль;
- **FastAPI** (≥0.104.0) — веб-фреймворк (HW04);
- **uvicorn** (≥0.24.0) — ASGI сервер (HW04);
- **pydantic** (≥2.0.0) — валидация данных в FastAPI (HW04).

**Для разработки:**

- **pytest** (≥7.0.0) — тестирование;
- **requests** (≥2.31.0) — HTTP клиент для тестирования (Вариант E).

Все зависимости управляются через `uv` и зафиксированы в `uv.lock`.

---

## 12. Примеры типичного сценария работы

### Сценарий 1: Полное тестирование всех компонентов

```bash
cd homeworks/HW04/eda-cli

# 1. Установка зависимостей
uv sync

# 2. Запуск тестов EDA-ядра
uv run pytest -q

# 3. Запуск CLI команд
uv run eda-cli overview data/example.csv
uv run eda-cli report data/example.csv --out-dir reports_final
uv run eda-cli head data/example.csv --n 5
uv run eda-cli sample data/example.csv --n 3

# 4. Запуск HTTP сервера (в отдельном терминале)
uv run uvicorn eda_cli.api:app --reload --port 8001

# 5. Тестирование API через браузер
# Откройте http://127.0.0.1:8001/docs

# 6. Или тестирование через клиент (в третьем терминале)
uv run python scripts/client.py

# 7. Проверка логов
cat logs/api.log | head -5
```

### Сценарий 2: Быстрое тестирование API через cURL

```bash
# Health-check
curl http://127.0.0.1:8001/health

# POST /quality
curl -X POST http://127.0.0.1:8001/quality \
  -H "Content-Type: application/json" \
  -d '{"n_rows": 2000, "n_cols": 15, "max_missing_share": 0.1, "numeric_cols": 8, "categorical_cols": 7}'

# POST /quality-from-csv
curl -X POST http://127.0.0.1:8001/quality-from-csv \
  -F "file=@data/example.csv"

# POST /quality-flags-from-csv
curl -X POST http://127.0.0.1:8001/quality-flags-from-csv \
  -F "file=@data/example.csv"

# GET /metrics
curl http://127.0.0.1:8001/metrics
```

### Сценарий 3: Анализ логов

```bash
# Просмотр всех логов
tail -10 logs/api.log

# Поиск запросов к quality-from-csv
grep "quality-from-csv" logs/api.log

# Подсчёт количества ошибок (если они есть)
grep '"status": 40' logs/api.log | wc -l

# Красивый вывод одного лога
cat logs/api.log | head -1 | python -m json.tool
```

---

## 13. Структура модулей

### `src/eda_cli/core.py`

Основная логика анализа (из HW03, без изменений):

- `load_csv()` — загрузка CSV;
- `get_basic_stats()` — базовая статистика;
- `get_missing_info()` — информация о пропусках;
- `compute_quality_flags()` — вычисление флагов качества;
- `get_problematic_columns()` — список колонок с высокой долей пропусков.

### `src/eda_cli/viz.py`

Визуализация (из HW03, без изменений):

- `save_histograms()` — гистограммы;
- `save_missing_bar()` — график пропусков;
- `save_boxplots()` — boxplot-графики;
- `save_category_bar()` — bar-chart для категориального признака.

### `src/eda_cli/cli.py`

CLI интерфейс на typer (из HW03, без изменений):

- `overview()` — команда обзора;
- `report()` — команда генерации отчёта;
- `head()` — команда вывода первых N строк;
- `sample()` — команда вывода случайной выборки.

### `src/eda_cli/api.py` (← НОВОЕ в HW04)

HTTP API на FastAPI:

**Настройка логирования (Вариант D):**

- `log_request()` — функция логирования в JSON формат;
- автоматическое создание папки `logs/` и файла `api.log`.

**Модели данных (Pydantic):**

- `QualityRequest` — структура для запроса `/quality`;
- `QualityResponse` — структура ответа;
- `QualityFlagsResponse` — структура ответа для полных флагов (HW04).

**Эндпоинты:**

- `GET /health` — health-check;
- `POST /quality` — эвристическая оценка по числовым признакам;
- `POST /quality-from-csv` — реальная оценка через загрузку файла;
- `POST /quality-flags-from-csv` — полный набор флагов (HW04);
- `GET /metrics` — метрики сервиса (Вариант F).

**Обработка запросов:**

- Конвертация numpy типов в нативные Python типы (`convert_to_native_types`);
- Расчёт latency для каждого запроса;
- Структурированное логирование в JSON;
- Обработка ошибок с HTTP статусами.

### `scripts/client.py` (← НОВОЕ в HW04, Вариант E)

Клиентский скрипт для тестирования API:

- Тестирование всех эндпоинтов;
- Красивый вывод результатов через `rich`;
- Сводная таблица по тестам `/quality`.

### `tests/test_core.py`

Набор модульных тестов pytest (из HW03, без изменений):

- Тесты для функций из `core.py`;
- 19 тестов, все проходят успешно.

---

## 14. Варианты решения

### Вариант D: Структурированное логирование

✅ **Реализовано в полном объёме.**

- Каждый запрос логируется в `logs/api.log` в JSON-формате;
- Логирование с информацией: timestamp, request_id, endpoint, status, latency_ms и дополнительные поля;
- Одновременный вывод логов в консоль (с префиксом `[INFO]`) и в файл;
- Запись файла в кодировке UTF-8 для корректной работы с русским текстом.

**Пример логов из `logs/api.log`:**

```json
{"timestamp": "2025-12-21T17:06:35.341133", "request_id": "149602da-dc90-4bdb-bd7d-b36061036a8f", "endpoint": "quality", "status": 200, "latency_ms": 0.01, "n_rows": 2000, "n_cols": 15, "ok_for_model": true, "quality_score": 0.8}
{"timestamp": "2025-12-21T17:07:29.845095", "request_id": "95001c5a-57f9-41c1-8c52-fcbb9b442475", "endpoint": "quality-from-csv", "status": 200, "latency_ms": 42.95, "filename": "example.csv", "n_rows": 15, "n_cols": 7, "ok_for_model": true, "quality_score": 1.0}
```

---

### Вариант E: Клиентский скрипт

✅ **Реализовано (опционально).**

- Расположен в `scripts/client.py`;
- Автоматически тестирует все эндпоинты API;
- Проверяет health-check, POST запросы с JSON и multipart/form-data;
- Выводит красивую таблицу результатов через `rich`;
- Удобен для быстрой проверки работоспособности API.

**Запуск:**

```bash
uv run python scripts/client.py
```

---

### Вариант F: Эндпоинт /metrics

✅ **Реализовано в полном объёме.**

- Эндпоинт `GET /metrics` возвращает статистику работы сервиса;
- Счётчики: общее количество запросов, средняя задержка, вызовы по эндпоинтам, ошибки;
- Отслеживание последнего значения флага `ok_for_model`;
- Метрики обновляются в реальном времени при каждом запросе.

**Пример ответа:**

```json
{
  "total_requests": 4,
  "avg_latency_ms": 12.44,
  "endpoint_calls": {
    "quality": 1,
    "quality-from-csv": 1,
    "quality-flags-from-csv": 1
  },
  "last_ok_for_model": true,
  "errors": 0
}
```

---

## 15. Проверка работоспособности

Используйте этот чеклист для полной проверки:

```bash
cd homeworks/HW04/eda-cli

# 1. Установка
uv sync

# 2. Тесты EDA-ядра
uv run pytest -q
# Ожидание: 19 passed

# 3. CLI команды
uv run eda-cli overview data/example.csv
uv run eda-cli report data/example.csv --out-dir reports_final
uv run eda-cli head data/example.csv --n 5
uv run eda-cli sample data/example.csv --n 3
# Все должны выполниться без ошибок

# 4. Запуск HTTP сервера
uv run uvicorn eda_cli.api:app --reload --port 8001
# В браузере: http://127.0.0.1:8001/docs

# 5. Проверка логирования
cat logs/api.log
# Должны быть JSON логи

# 6. Проверка метрик
curl http://127.0.0.1:8001/metrics
# Должен вернуть JSON с метриками
```

---

## 16. Контрольный список (Вариант D, E, F)

- [x] **D. Структурированное логирование**
  - [x] Логирование всех запросов в `logs/api.log`
  - [x] JSON-формат с timestamp, request_id, endpoint, status, latency_ms
  - [x] Дополнительные поля в зависимости от типа запроса
  - [x] Вывод также в консоль с префиксом `[INFO]`

- [x] **E. Клиентский скрипт (опционально)**
  - [x] `scripts/client.py` для тестирования всех эндпоинтов
  - [x] Красивый вывод через `rich`
  - [x] Сводная таблица результатов

- [x] **F. Эндпоинт /metrics**
  - [x] Возвращает статистику работы сервиса
  - [x] total_requests, avg_latency_ms, endpoint_calls
  - [x] last_ok_for_model, errors
  - [x] Обновляется в реальном времени

---

## 17. Дополнительная информация

### Интеграция с HW03

Проект полностью сохраняет функциональность HW03:

- все CLI команды работают без изменений;
- EDA-ядро (`core.py`) переиспользуется в API;
- тесты всё ещё проходят без изменений (19 passed).

### Безопасность

- API валидирует входные данные через Pydantic;
- Неверные типы данных и форматы файлов отклоняются с HTTP 400;
- Обработка исключений с informative error messages.

### Масштабируемость

- Структура позволяет легко добавлять новые эндпоинты;
- Логирование фиксирует всю историю запросов для аналитики;
- Метрики помогают отслеживать производительность;
- EDA-ядро можно переиспользовать в других проектах.

---

## 18. Команды для быстрого старта

```bash
# Клонирование и установка
cd homeworks/HW04/eda-cli
uv sync

# Запуск всех тестов
uv run pytest -q

# Запуск HTTP сервера
uv run uvicorn eda_cli.api:app --reload --port 8001

# В другом терминале: тестирование клиентом
uv run python scripts/client.py

# Или вручную через браузер
# http://127.0.0.1:8001/docs

# Или через cURL
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/metrics
```

---

## 19. Заключение

HW04 успешно расширяет EDA CLI из HW03 с добавлением полнофункционального HTTP API на FastAPI. Проект демонстрирует:

✅ Качественную интеграцию EDA-ядра с веб-фреймворком  
✅ Структурированное логирование для production-среды  
✅ Метрики и статистику для мониторинга  
✅ Полное покрытие тестами  
✅ Красивую интерактивную документацию (Swagger UI)  

Все компоненты работают и готовы к использованию!