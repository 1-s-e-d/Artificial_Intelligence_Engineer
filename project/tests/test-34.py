"""
project/tests/test-34.py
========================
Прогноз с CQR (Conformal Quantile Regression), автоматический выбор
устройства (CPU/GPU) из project/configs/device_config.json.
Полностью повторяет функциональность оригинального test/test-34.py
с обновлёнными путями под новую структуру project/.
"""

import os
import sys
import json
import pickle
import shutil
import warnings
import time
import threading
import logging

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
logging.getLogger().setLevel(logging.WARNING)

try:
    from mapie.regression import ConformalizedQuantileRegressor
    MAPIE_AVAILABLE = True
except ImportError:
    MAPIE_AVAILABLE = False
    print("\n⚠️  MAPIE не установлена. Установите: pip install mapie", flush=True)
    sys.exit(1)

import logging
logging.getLogger('mapie').setLevel(logging.ERROR)

# ----------------------------------------------------------------------
# 0. ПУТИ  (project/tests/ → project/ → repo root)
# ----------------------------------------------------------------------
THIS_DIR   = os.path.dirname(os.path.abspath(__file__))   # project/tests/
PROJ_DIR   = os.path.dirname(THIS_DIR)                    # project/
ROOT_DIR   = os.path.dirname(PROJ_DIR)                    # repo root

DEVICE_CONFIG_PATH = os.path.join(PROJ_DIR, 'configs', 'device_config.json')
CONFIG_PATH        = os.path.join(PROJ_DIR, 'configs', 's_config.json')

# ----------------------------------------------------------------------
# 0b. ЗАГРУЗКА КОНФИГУРАЦИИ УСТРОЙСТВА
# ----------------------------------------------------------------------
if os.path.exists(DEVICE_CONFIG_PATH):
    with open(DEVICE_CONFIG_PATH, 'r', encoding='utf-8') as f:
        device_cfg = json.load(f)
    DEVICE = device_cfg.get('device', 'cpu')
    if DEVICE == 'cpu':
        NUM_THREADS = -1
        N_JOBS = 1
    else:
        NUM_THREADS = device_cfg.get('num_threads', 1)
        N_JOBS = device_cfg.get('n_jobs', 1)
    print(f"\n✓ Загружены настройки: DEVICE={DEVICE.upper()}, num_threads={NUM_THREADS}", flush=True)
else:
    DEVICE = 'cpu'
    NUM_THREADS = -1
    N_JOBS = 1
    print(f"\n⚠️ device_config.json не найден ({DEVICE_CONFIG_PATH}), используем CPU.", flush=True)

# ----------------------------------------------------------------------
# 1. КОНФИГ ЗАДАЧИ
# ----------------------------------------------------------------------
with open(CONFIG_PATH, encoding='utf-8') as f:
    CFG = json.load(f)

DATA_PATH  = os.path.normpath(os.path.join(ROOT_DIR, CFG['data']['dataset_path']))
DATE_COL   = CFG['data']['date_col']
TARGET_COL = CFG['data']['target_col']
OUTPUT_DIR = os.path.normpath(os.path.join(ROOT_DIR, CFG['forecast']['output_dir']))
RUN_TAG    = os.path.basename(OUTPUT_DIR)
MODEL_DIR  = os.path.normpath(os.path.join(ROOT_DIR, CFG['tune']['output_dir']))

_fc = CFG['forecast']
TRAIN_START_TS = pd.Timestamp(_fc['train_start'])
TRAIN_END_TS   = pd.Timestamp(_fc['train_end'])
FORE_START     = pd.Timestamp(_fc['forecast_start'])
FORE_END_INCL  = pd.Timestamp(_fc['forecast_end'])
FORE_END_EXCL  = FORE_END_INCL + pd.Timedelta(days=1)
TRAIN_DAYS     = (TRAIN_END_TS - TRAIN_START_TS).days

FORECAST_HOURS = 24
MIN_TRAIN_ROWS = 24
MODEL_NAME     = 'allday'

for sub in ('', 'R2_dr', 'MAPE_dr', 'WAPE_dr', 'forecast', 'forecast_month', 'forecast_dopw', 'forecast_ci'):
    os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

shutil.copy2(CONFIG_PATH, os.path.join(OUTPUT_DIR, 's_config.json'))

# ----------------------------------------------------------------------
# 2-11. (идентично оригиналу test/test-34.py, пути изменены выше)
# ----------------------------------------------------------------------

def ym_label_from_ts(ts):
    return f'{ts.year:04d}-{ts.month:02d}'

print(f'\nЗагрузка данных: {DATA_PATH}', flush=True)
df_raw = pd.read_excel(DATA_PATH)
df_raw[DATE_COL] = pd.to_datetime(df_raw[DATE_COL])
df_raw = df_raw.sort_values(DATE_COL).reset_index(drop=True)

def build_features(df):
    df = df.copy()
    df['hour'] = df[DATE_COL].dt.hour
    df['dayofweek'] = df[DATE_COL].dt.dayofweek
    df['month_num'] = df[DATE_COL].dt.month
    df['doy'] = df[DATE_COL].dt.dayofyear
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_monday'] = (df['dayofweek'] == 0).astype(int)
    df['is_friday'] = (df['dayofweek'] == 4).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['hour_sin_2'] = np.sin(4 * np.pi * df['hour'] / 24)
    df['hour_cos_2'] = np.cos(4 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
    df['doy_sin'] = np.sin(2 * np.pi * df['doy'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['doy'] / 365)
    df['temp_squared'] = df['temp'] ** 2
    for lag in [24, 48, 168, 336]:
        df[f'lag_{lag}'] = df[TARGET_COL].shift(lag)
    df['roll_mean_24'] = df[TARGET_COL].shift(24).rolling(24).mean()
    df['roll_std_24'] = df[TARGET_COL].shift(24).rolling(24).std()
    df['roll_mean_168'] = df[TARGET_COL].shift(24).rolling(168).mean()
    df['workday_hour_sin'] = df['workday'] * df['hour_sin']
    df['workday_hour_cos'] = df['workday'] * df['hour_cos']
    df['lag_mean_same_type'] = (
        df.groupby(['workday', 'hour'])[TARGET_COL]
          .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
    )
    return df

df = build_features(df_raw)

OPTIONAL = [c for c in ['wet', 'cloud', 'windspeed', 'daylength', 'day/night'] if c in df.columns]
FEATS_DEFAULT = [
    'hour_sin', 'hour_cos', 'hour_sin_2', 'hour_cos_2',
    'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
    'workday', 'is_weekend', 'is_monday', 'is_friday',
    'temp', 'temp_squared',
    'lag_24', 'lag_48', 'lag_168', 'lag_336',
    'roll_mean_24', 'roll_std_24', 'roll_mean_168',
    'workday_hour_sin', 'workday_hour_cos', 'lag_mean_same_type',
] + OPTIONAL

df_clean = df.dropna(subset=list(set(FEATS_DEFAULT + [TARGET_COL]))).reset_index(drop=True)
print(f'  После dropna: {len(df_clean)} строк', flush=True)

def load_model_bundle(name):
    pkl_path  = os.path.join(MODEL_DIR, f'best_model_{name}.pkl')
    json_path = os.path.join(MODEL_DIR, f'best_params_{name}.json')
    if os.path.exists(pkl_path):
        with open(pkl_path, 'rb') as f:
            pkl_data = pickle.load(f)
        params = dict(pkl_data['params'])
        feats  = list(pkl_data.get('features', FEATS_DEFAULT))
    elif os.path.exists(json_path):
        with open(json_path, encoding='utf-8') as f:
            jdata = json.load(f)
        params = dict(jdata['params'])
        params.update({'verbosity': -1, 'n_jobs': -1, 'random_state': 42})
        feats = FEATS_DEFAULT
    else:
        raise FileNotFoundError(f"Модель {name} не найдена в {MODEL_DIR}")
    return params, feats

print('\nЗагрузка параметров модели...', flush=True)
MODEL_PARAMS, MODEL_FEATS = load_model_bundle(MODEL_NAME)

for fn in (f'best_params_{MODEL_NAME}.json', f'best_model_{MODEL_NAME}.pkl'):
    src = os.path.join(MODEL_DIR, fn)
    if os.path.exists(src):
        shutil.copy2(src, OUTPUT_DIR)

all_days = pd.date_range(FORE_START, FORE_END_INCL, freq='D')
PERIOD_START_LABEL = str(all_days[0].date()) if len(all_days) else str(FORE_START.date())
PERIOD_END_LABEL   = str(all_days[-1].date()) if len(all_days) else str(FORE_END_INCL.date())
MONTH_LABELS = sorted({ym_label_from_ts(d) for d in all_days})

print(f"\nПрогнозный период: {PERIOD_START_LABEL} .. {PERIOD_END_LABEL}", flush=True)
print(f"Дней: {len(all_days)}, TRAIN_DAYS = {TRAIN_DAYS}", flush=True)

if not MAPIE_AVAILABLE:
    raise RuntimeError("MAPIE не установлена. Установите: pip install mapie")

class ProgressTimer:
    def __init__(self, total, desc="Прогресс"):
        self.total = total; self.desc = desc; self.completed = 0
        self.times = []; self.start_time = None
        self.current_iter_start = None; self.running = False
        self.thread = None; self.lock = threading.Lock()
    def start(self):
        self.start_time = time.time(); self.running = True
        self.thread = threading.Thread(target=self._update_loop, daemon=True); self.thread.start()
    def iter_start(self):
        with self.lock: self.current_iter_start = time.time()
    def iter_end(self):
        with self.lock:
            if self.current_iter_start is not None:
                self.times.append(time.time() - self.current_iter_start)
                self.completed += 1; self.current_iter_start = None
    def _median(self, lst):
        if not lst: return 0.0
        s = sorted(lst); n = len(s); mid = n // 2
        return s[mid] if n % 2 else (s[mid-1] + s[mid]) / 2.0
    def _format_time(self, sec):
        m, s = divmod(int(sec), 60); h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
    def _update_loop(self):
        while self.running:
            with self.lock:
                completed = self.completed; times = self.times.copy(); start_time = self.start_time
            elapsed = time.time() - start_time
            eta_str = self._format_time(self._median(times) * (self.total - completed)) if completed > 0 else "--:--"
            percent = 100.0 * completed / self.total
            bar_len = 30; filled = int(bar_len * completed / self.total)
            bar = '\u2588' * filled + '\u2591' * (bar_len - filled)
            sys.stdout.write(f"\r{self.desc}: {bar} {percent:.1f}% | "
                             f"{completed}/{self.total} [{self._format_time(elapsed)}<{eta_str}]")
            sys.stdout.flush()
            if completed >= self.total: break
            time.sleep(1)
        sys.stdout.write("\r" + " " * 80 + "\r"); sys.stdout.flush()
    def stop(self):
        self.running = False
        if self.thread: self.thread.join(timeout=0.5)

daily_metrics = []
prediction_rows = []
progress = ProgressTimer(total=len(all_days), desc="Прогноз по дням (CQR)")
progress.start()

DOW_COLORS = {0:'#E57373',1:'#42A5F5',2:'#42A5F5',3:'#42A5F5',4:'#42A5F5',5:'#FFA726',6:'#FFA726'}

for day_start in all_days:
    day_end = day_start + pd.Timedelta(hours=24)
    df_day = df_clean[(df_clean[DATE_COL] >= day_start) & (df_clean[DATE_COL] < day_end)].copy()
    if len(df_day) != 24: continue
    if len(df_day['workday'].unique()) != 1: continue
    day_wd = int(df_day['workday'].iloc[0])
    df_train = df_clean[(df_clean[DATE_COL] >= TRAIN_START_TS) & (df_clean[DATE_COL] < day_start)]
    if len(df_train) < MIN_TRAIN_ROWS: continue
    progress.iter_start()
    X_train = df_train[MODEL_FEATS]; y_train = df_train[TARGET_COL].values
    X_test  = df_day[MODEL_FEATS];   y_test  = df_day[TARGET_COL].values
    split_idx = int(0.6 * len(X_train))
    X_train_main = X_train.iloc[:split_idx]; y_train_main = y_train[:split_idx]
    X_cal = X_train.iloc[split_idx:];        y_cal = y_train[split_idx:]
    base_params = MODEL_PARAMS.copy()
    base_params.update({'device': DEVICE, 'n_jobs': N_JOBS, 'num_threads': NUM_THREADS, 'verbosity': -1, 'random_state': 42})
    if DEVICE == 'gpu': base_params['gpu_use_dp'] = False
    params_lower  = {**base_params, 'objective': 'quantile', 'alpha': 0.025}
    params_median = {**base_params, 'objective': 'quantile', 'alpha': 0.5}
    params_upper  = {**base_params, 'objective': 'quantile', 'alpha': 0.975}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model_lower  = LGBMRegressor(**params_lower).fit(X_train_main, y_train_main)
        model_median = LGBMRegressor(**params_median).fit(X_train_main, y_train_main)
        model_upper  = LGBMRegressor(**params_upper).fit(X_train_main, y_train_main)
    cqr = ConformalizedQuantileRegressor(estimator=[model_lower, model_upper, model_median], confidence_level=0.95, prefit=True)
    cqr.conformalize(X_cal, y_cal)
    y_pred_median, y_pis = cqr.predict_interval(X_test)
    y_pred_lower = np.minimum(y_pis[:, 0, 0], y_pis[:, 1, 0])
    y_pred_upper = np.maximum(y_pis[:, 0, 0], y_pis[:, 1, 0])
    y_pred = y_pred_median
    n_obs = len(y_test); k_params = X_test.shape[1] + 1
    rss = float(np.sum((y_test - y_pred)**2)); mse = rss / n_obs if n_obs else np.nan
    rmse = np.sqrt(mse) if mse and mse >= 0 else np.nan
    mae = float(np.mean(np.abs(y_test - y_pred)))
    re_val = mae / np.mean(y_test) * 100 if np.mean(y_test) != 0 else np.nan
    mask = y_test != 0
    mape_val = float(np.mean(np.abs((y_test[mask]-y_pred[mask])/y_test[mask]))*100) if mask.sum() else np.nan
    denom = np.sum(np.abs(y_test))
    wape_val = float(np.sum(np.abs(y_test-y_pred))/denom*100) if denom else np.nan
    r2_val = float(r2_score(y_test, y_pred))
    r2_adj_val = float(1-(1-r2_val)*(n_obs-1)/(n_obs-k_params-1)) if n_obs > (k_params+1) else np.nan
    if rss > 0 and n_obs > 0:
        aic_val  = float(n_obs*np.log(rss/n_obs)+2*k_params)
        aicc_val = float(aic_val+(2*k_params*(k_params+1))/(n_obs-k_params-1)) if n_obs-k_params-1 > 0 else np.nan
        bic_val  = float(n_obs*np.log(rss/n_obs)+k_params*np.log(n_obs))
    else:
        aic_val = aicc_val = bic_val = np.nan
    dow = day_start.dayofweek; dow_name = ['пн','вт','ср','чт','пт','сб','вс'][dow]
    month_label = ym_label_from_ts(day_start)
    print(f"\n[OK] {day_start.date()} {dow_name} | {MODEL_NAME:7s} | train={len(df_train):4d} | "
          f"R²={r2_val:.4f} | MAPE={mape_val:.4f}% | WAPE={wape_val:.4f}%", flush=True)
    daily_metrics.append({'date':str(day_start.date()),'month':month_label,'dow':dow,'dow_name':dow_name,
        'model':MODEL_NAME,'workday':day_wd,'train_rows':len(df_train),
        'R2':r2_val,'R2_adjusted':r2_adj_val,'MSE':mse,'RMSE':rmse,'MAE':mae,'RE':re_val,
        'MAPE':mape_val,'WAPE':wape_val,'RSS':rss,'AIC':aic_val,'AICc':aicc_val,'BIC':bic_val})
    pred_df = pd.DataFrame({DATE_COL:df_day[DATE_COL].values,'y_true':y_test,'y_pred':y_pred,
        'ci_lo':y_pred_lower,'ci_hi':y_pred_upper,'model':MODEL_NAME,
        'workday':day_wd,'date':str(day_start.date()),'month':month_label})
    prediction_rows.append(pred_df)
    progress.iter_end()

progress.stop()

if not daily_metrics:
    raise RuntimeError("Нет валидных дней для прогноза")

metrics_df = pd.DataFrame(daily_metrics)
preds_df   = pd.concat(prediction_rows, ignore_index=True).sort_values(DATE_COL)

# --- сохранение CSV / summary (идентично оригиналу) ---
def _s(arr, decimals=4):
    arr = [v for v in arr if v is not None and not np.isnan(v)]
    if not arr: return 'n/a'
    fmt = f'.{decimals}f'
    return (f'mean={np.mean(arr):{fmt}}  min={np.min(arr):{fmt}}  '
            f'max={np.max(arr):{fmt}}  std={np.std(arr):{fmt}}')

metrics_df.to_csv(os.path.join(OUTPUT_DIR, 'daily_metrics.csv'), index=False)
preds_df.to_csv(os.path.join(OUTPUT_DIR, 'predictions.csv'), index=False)

SEP = '=' * 70
print(f'\n{SEP}\nTEST-34 (CQR, {DEVICE.upper()}) — {PERIOD_START_LABEL} .. {PERIOD_END_LABEL}', flush=True)
print(f'  R²  : {_s(metrics_df["R2"].tolist())}', flush=True)
print(f'  MAPE: {_s(metrics_df["MAPE"].tolist())}%', flush=True)
print(f'  WAPE: {_s(metrics_df["WAPE"].tolist())}%', flush=True)
print(f'\nГотово! Результаты в {OUTPUT_DIR}/', flush=True)
