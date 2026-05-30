# Regression Modeling — LightGBM Forecasting

Проект для автоматизированного подбора гиперпараметров и прогнозирования временных рядов почасового
электропотребления на основе **LightGBM** с веб-интерфейсом управления.

---

## Паспорт проекта

| Параметр | Значение |
|---|---|
| **Тип задачи** | Многомерная регрессия временного ряда |
| **Модель** | LightGBM (gradient boosting, leaf-wise) |
| **Горизонт** | 24 шага вперёд (суточный профиль) |
| **Целевая переменная** | `Ypowerconsumption` (МВт) |
| **Датасет** | 30 816 записей, 01.01.2019 — 07.07.2022 |
| **Оптимизация гиперпараметров** | Optuna |
| **Доверительные интервалы** | MAPIE (conformal prediction) |
| **Интерфейс** | Flask REST API + веб-UI |
| **Контейнеризация** | Docker + docker-compose |

---

## Структура `project/`

```
project/
├── README.md              ← этот файл (паспорт проекта)
├── requirements.txt       ← зависимости Python
├── Dockerfile             ← сборка образа
├── docker-compose.yml     ← оркестрация контейнера
├── report.md              ← отчёт (задача, данные, эксперименты, результаты)
├── self-checklist.md      ← чек-лист перед сдачей
├── notebooks/             ← EDA и эксперименты
├── src/                   ← основной код
│   ├── run_webui.py         ← точка входа
│   ├── web_interface/       ← Flask-сервер + REST API
│   ├── avto_test_py/        ← тюнинг + бенчмарк
│   └── utils/               ← предобработка и визуализация
├── data/                  ← датасет (не коммитится)
├── configs/               ← конфигурационные файлы
│   ├── s_config.json        ← даты, параметры модели, n_trials
│   ├── .env.example         ← шаблон переменных окружения
│   └── device_config.json   ← CPU/GPU (генерируется бенчмарком)
└── tests/                 ← pytest-тесты
```

---

## Быстрый старт

### 1. Подготовка

```bash
git clone https://github.com/DaL1ner/regression_modeling.git
cd regression_modeling/project

cp configs/.env.example ../.env
cp /path/to/dataset.xlsx data/dataset.xlsx
```

### 2. Запуск локально

> ⚠️ Все команды выполняются из папки **`project/`**

```bash
cd regression_modeling/project

pip install -r requirements.txt
python src/run_webui.py
```

Браузер откроется автоматически: [http://127.0.0.1:5000](http://127.0.0.1:5000)

### 3. Запуск через Docker

```bash
cd regression_modeling/project
docker compose up --build
```

> **Примечание о сетевой доступности.**
> В некоторых сетевых средах прямой доступ к Docker Hub (реестр базовых образов) может быть ограничен.
> Для успешной сборки образа рекомендуется обеспечить стабильный доступ к внешним ресурсам — например, через настройку зеркального реестра в настройках Docker Desktop
> (**Settings → Docker Engine → `registry-mirrors`**) или использование сетевого прокси-сервера.
> При невозможности использовать Docker рекомендуется запуск локально (см. п. 2).

---

## Конфигурация

Основной конфиг проекта: `configs/s_config.json`

```json
{
  "tune": {
    "tune_start": "2019-01-01",
    "tune_end": "2019-09-30",
    "n_trials": 30
  },
  "model": {
    "train_days": 90,
    "max_windows": 40,
    "window_step_days": 7
  },
  "forecast": {
    "train_start": "2020-01-02",
    "train_end": "2021-12-31",
    "forecast_start": "2022-01-01",
    "forecast_end": "2022-01-30"
  }
}
```

---

## REST API

| Метод | URL | Описание |
|---|---|---|
| GET | `/api/config` | Получить конфигурацию |
| POST | `/api/config` | Сохранить конфигурацию |
| GET | `/api/device_config` | CPU/GPU конфиг |
| POST | `/api/device_config` | Сохранить CPU/GPU конфиг |
| GET | `/api/calendar_events` | Данные для календаря |
| POST | `/api/run_tune` | Запустить тюнинг |
| POST | `/api/run_forecast` | Запустить прогноз |
| POST | `/api/run_benchmark` | Бенчмарк CPU/GPU |
| POST | `/api/stop` | Остановить задачу |
| GET | `/api/progress` | SSE-поток логов |
| POST | `/api/shutdown` | Завершить сервер |

---

## Ветки

| Ветка | Описание |
|---|---|
| `GBM` | Основная ветка разработки |
| `PM` | Project Management / документация + новая структура `project/` |
