# Self-checklist — проверка перед сдачей

## Данные
- [ ] `project/data/dataset.xlsx` присутствует локально и соответствует схеме (30 816 строк, 13 колонок)
- [ ] Данные не попали в репозиторий (`.gitignore` закрывает `project/data/*.xlsx`)
- [ ] Все признаки (`hour`, `temp`, `daylength`, `cloud`, `wet`, `winddir`, `windspeed`, `sunrise`, `sunset`, `day/night`, `workday`) присутствуют в датасете
- [ ] Целевая переменная `Ypowerconsumption` без пропусков после предфильтрации

## Конфигурация
- [ ] `project/configs/s_config.json` заполнен корректными датами (`tune_start`, `tune_end`, `train_start`, `train_end`, `forecast_start`, `forecast_end`)
- [ ] `project/configs/.env.example` скопирован в `.env` и заполнен (не закоммичен)
- [ ] `FLASK_PORT` не занят другим процессом
- [ ] Если используется GPU — `project/configs/device_config.json` существует и содержит корректный `device`

## Среда выполнения
- [ ] Python ≥ 3.11
- [ ] Все зависимости установлены: `pip install -r requirements.txt`
- [ ] `lightgbm` компилируется с OpenMP (`lgb.train` не падает на тестовом вызове)
- [ ] При запуске через Docker: `docker compose up --build` завершается без ошибок
- [ ] При локальном запуске: `python project/src/run_webui.py` открывает браузер на `http://127.0.0.1:5000`

## Бенчмарк
- [ ] `/api/run_benchmark` завершается успешно и создаёт `project/configs/device_config.json`
- [ ] Кириллица в логах бенчмарка отображается корректно (нет кракозябр)

## Тюнинг
- [ ] `/api/run_tune` запускается без `409 Conflict`
- [ ] Optuna выполняет ≥ 1 триал без исключений
- [ ] Артефакты тюнинга сохраняются в `artefakts/`

## Прогноз
- [ ] `/api/run_forecast` запускается после тюнинга
- [ ] CSV с прогнозом появляется в `artefakts/`
- [ ] MAPE ≤ 10%, WAPE ≤ 8%, R² ≥ 0.85 на тестовом периоде

## Код
- [ ] Нет захардкоженных абсолютных путей (все пути через `os.path` относительно `__file__` или переменных окружения)
- [ ] Новая структура `project/` используется как основная
- [ ] Прокси-обёртки в корне работают (обратная совместимость)
- [ ] `project/tests/` содержит хотя бы smoke-тест

## Документация
- [ ] `project/README.md` актуален (описание, установка, запуск)
- [ ] `project/report.md` содержит описание экспериментов и результатов
- [ ] `project/notebooks/` содержит EDA-ноутбук с основными визуализациями
