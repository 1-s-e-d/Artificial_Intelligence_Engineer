"""
benchmark_device.py
===================
Автоматическое определение оптимального устройства (CPU/GPU) для LightGBM.
Запускается один раз перед использованием test-34.py.

Алгоритм:
1. Загружает данные и создаёт признаки (как в основном скрипте).
2. Берёт последние 3000 строк (для скорости).
3. Обучает модель с фиксированными гиперпараметрами на CPU с максимальной многопоточностью.
4. Обучает ту же модель на GPU с оптимальными настройками.
5. Сравнивает время и выбирает лучшее устройство.
6. Сохраняет конфиг в project/configs/device_config.json.

Запуск (из корня репозитория):
    python project/src/avto_test_py/benchmark_device.py
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

# ----------------------------------------------------------------------
# Пути
# project/src/avto_test_py/ -> project/src/ -> project/ -> ROOT
# ----------------------------------------------------------------------
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.dirname(THIS_DIR)
PROJ_DIR   = os.path.dirname(SRC_DIR)
ROOT_DIR   = os.path.dirname(PROJ_DIR)

CONFIG_PATH        = os.path.join(PROJ_DIR, 'configs', 's_config.json')
DEVICE_CONFIG_PATH = os.path.join(PROJ_DIR, 'configs', 'device_config.json')

if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Конфиг не найден: {CONFIG_PATH}")

with open(CONFIG_PATH, encoding='utf-8') as f:
    CFG = json.load(f)

DATA_PATH  = os.path.normpath(os.path.join(ROOT_DIR, CFG['data']['dataset_path']))
DATE_COL   = CFG['data']['date_col']
TARGET_COL = CFG['data']['target_col']

print(f"Загрузка данных: {DATA_PATH}", flush=True)
df_raw = pd.read_excel(DATA_PATH)
df_raw[DATE_COL] = pd.to_datetime(df_raw[DATE_COL])
df_raw = df_raw.sort_values(DATE_COL).reset_index(drop=True)

# ----------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------
def build_features_benchmark(df):
    df = df.copy()
    df['hour']      = df[DATE_COL].dt.hour
    df['dayofweek'] = df[DATE_COL].dt.dayofweek
    df['month_num'] = df[DATE_COL].dt.month
    df['doy']       = df[DATE_COL].dt.dayofyear
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_monday']  = (df['dayofweek'] == 0).astype(int)
    df['is_friday']  = (df['dayofweek'] == 4).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin']  = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos']  = np.cos(2 * np.pi * df['dayofweek'] / 7)
    for lag in [24, 48, 168, 336]:
        df[f'lag_{lag}'] = df[TARGET_COL].shift(lag)
    df['roll_mean_24']  = df[TARGET_COL].shift(24).rolling(24).mean()
    df['roll_std_24']   = df[TARGET_COL].shift(24).rolling(24).std()
    df['roll_mean_168'] = df[TARGET_COL].shift(24).rolling(168).mean()
    if 'workday' not in df.columns:
        df['workday'] = 0
    return df

print("Построение признаков...", flush=True)
df = build_features_benchmark(df_raw)

FEATURES = [
    'hour_sin', 'hour_cos',
    'dow_sin', 'dow_cos',
    'workday', 'is_weekend', 'is_monday', 'is_friday',
    'lag_24', 'lag_48', 'lag_168', 'lag_336',
    'roll_mean_24', 'roll_std_24', 'roll_mean_168'
]

df_clean = df.dropna(subset=FEATURES + [TARGET_COL]).reset_index(drop=True)
print(f"Строк после очистки: {len(df_clean)}", flush=True)

sample_size = min(3000, len(df_clean))
df_sample = df_clean.tail(sample_size).reset_index(drop=True)
X = df_sample[FEATURES]
y = df_sample[TARGET_COL].values

split = int(0.8 * len(X))
X_train, X_val = X.iloc[:split], X.iloc[split:]
y_train, y_val = y[:split], y[split:]

base_params = {
    'objective': 'quantile',
    'alpha': 0.5,
    'n_estimators': 100,
    'num_leaves': 63,
    'learning_rate': 0.05,
    'verbosity': -1,
    'random_state': 42,
    'early_stopping_round': 10,
}

# ----------------------------------------------------------------------
# CPU тест
# ----------------------------------------------------------------------
print("\n=== Тест CPU (все ядра) ===", flush=True)
cpu_params = {**base_params, 'device': 'cpu', 'n_jobs': 1, 'num_threads': -1}
model_cpu = LGBMRegressor(**cpu_params)
start = time.perf_counter()
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    model_cpu.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='mae')
time_cpu = time.perf_counter() - start
print(f"CPU время: {time_cpu:.2f} секунд", flush=True)

# ----------------------------------------------------------------------
# GPU тест
# ----------------------------------------------------------------------
print("\n=== Тест GPU ===", flush=True)
gpu_available = False
time_gpu = float('inf')
try:
    gpu_params = {**base_params, 'device': 'gpu', 'n_jobs': 1, 'num_threads': 1, 'gpu_use_dp': False}
    model_gpu = LGBMRegressor(**gpu_params)
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model_gpu.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='mae')
    time_gpu = time.perf_counter() - start
    gpu_available = True
    print(f"GPU время: {time_gpu:.2f} секунд", flush=True)
except Exception as e:
    print(f"GPU не доступен или ошибка: {e}", flush=True)
    print("Будет использован CPU.", flush=True)

# ----------------------------------------------------------------------
# Выбор устройства
# ----------------------------------------------------------------------
if gpu_available and time_gpu < time_cpu * 0.95:
    best_device = 'gpu'
    best_n_jobs = 1
    best_num_threads = 1
    reason = f"GPU быстрее: {time_gpu:.2f}с vs CPU {time_cpu:.2f}с"
else:
    best_device = 'cpu'
    best_n_jobs = 1
    best_num_threads = -1
    reason = f"CPU быстрее или GPU недоступен: CPU {time_cpu:.2f}с, GPU {time_gpu if gpu_available else 'N/A'}с"

print(f"\n--- Результат бенчмарка ---", flush=True)
print(f"Оптимальное устройство: {best_device.upper()}", flush=True)
print(f"Причина: {reason}", flush=True)

config = {
    'device': best_device,
    'n_jobs': best_n_jobs,
    'num_threads': best_num_threads,
    'benchmark_times_sec': {'cpu': round(time_cpu, 3), 'gpu': round(time_gpu, 3) if gpu_available else None},
    'comment': f'Автоопределение {time.strftime("%Y-%m-%d %H:%M:%S")}',
    'note': 'Для CPU используется num_threads=-1 (все ядра). Для GPU оптимизировано.'
}

os.makedirs(os.path.dirname(DEVICE_CONFIG_PATH), exist_ok=True)
with open(DEVICE_CONFIG_PATH, 'w', encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print(f"\nКонфигурация сохранена в: {DEVICE_CONFIG_PATH}", flush=True)
