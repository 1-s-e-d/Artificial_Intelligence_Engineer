"""
tune_workday_gpu.py  (project/src/avto_test_py/)
=================================================
Автоматический подбор гиперпараметров LightGBM — специалист РАБОЧИХ ДНЕЙ (workday == 1).
Читает конфиг из project/configs/s_config.json.

Запуск (из корня репозитория):
    python project/src/avto_test_py/tune_workday_gpu.py
"""

import os, sys, json, time, threading, warnings, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import optuna
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

print("Проверка GPU...", flush=True)
try:
    _t = LGBMRegressor(n_estimators=1, device='gpu', verbosity=-1)
    _t.fit([[1],[2],[3]], [1,2,3])
    print("  ✓ GPU доступен", flush=True)
except Exception as e:
    print(f"  ✗ GPU недоступен: {e}", flush=True)
    sys.exit(1)

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = os.path.dirname(os.path.dirname(THIS_DIR))
ROOT_DIR = os.path.dirname(PROJ_DIR)
sys.path.insert(0, ROOT_DIR)

CONFIG_PATH = os.path.join(PROJ_DIR, 'configs', 's_config.json')
with open(CONFIG_PATH, encoding='utf-8') as f:
    CFG = json.load(f)

DATA_PATH  = os.path.normpath(os.path.join(ROOT_DIR, CFG['data']['dataset_path']))
DATE_COL   = CFG['data']['date_col']
TARGET_COL = CFG['data']['target_col']

SPEC_CFG     = CFG['specialists']['workday']
SEARCH_SPACE = SPEC_CFG['lgbm_search_space']
TUNE_START       = pd.Timestamp(CFG['tune']['tune_start'])
TUNE_END         = pd.Timestamp(CFG['tune']['tune_end']) + pd.Timedelta(days=1)
TRAIN_DAYS       = int(SPEC_CFG['train_days'])
MAX_WINDOWS      = int(SPEC_CFG['max_windows'])
WINDOW_STEP_DAYS = int(SPEC_CFG.get('window_step_days', 1))
N_TRIALS         = int(CFG['tune']['n_trials'])
W_MAPE           = float(CFG['tune']['metric_mape_weight'])
W_R2             = float(CFG['tune']['metric_r2_weight'])
OUTPUT_DIR       = os.path.normpath(os.path.join(ROOT_DIR, CFG['tune']['output_dir']))

TRAIN_HOURS = TRAIN_DAYS * 24; FORECAST_HOURS = 24; STEP_HOURS = WINDOW_STEP_DAYS * 24
SPECIALIST = 'workday'; WORKDAY_FLAG = 1
os.makedirs(OUTPUT_DIR, exist_ok=True)

def suggest_from_space(trial, name, spec):
    t = spec['type']
    if t == 'int': return trial.suggest_int(name, int(spec['low']), int(spec['high']))
    return trial.suggest_float(name, float(spec['low']), float(spec['high']), log=bool(spec.get('log', False)))

print(f'\nЗагрузка данных: {DATA_PATH}', flush=True)
df = pd.read_excel(DATA_PATH)
df[DATE_COL] = pd.to_datetime(df[DATE_COL])
df = df.sort_values(DATE_COL).reset_index(drop=True)
df = df[(df[DATE_COL] >= TUNE_START) & (df[DATE_COL] < TUNE_END)].reset_index(drop=True)
print(f'  Строк: {len(df)}', flush=True)

df['hour'] = df[DATE_COL].dt.hour; df['dayofweek'] = df[DATE_COL].dt.dayofweek
df['month'] = df[DATE_COL].dt.month; df['doy'] = df[DATE_COL].dt.dayofyear
df['is_weekend']=(df['dayofweek']>=5).astype(int); df['is_monday']=(df['dayofweek']==0).astype(int)
df['is_friday']=(df['dayofweek']==4).astype(int)
df['hour_sin']=np.sin(2*np.pi*df['hour']/24); df['hour_cos']=np.cos(2*np.pi*df['hour']/24)
df['hour_sin_2']=np.sin(4*np.pi*df['hour']/24); df['hour_cos_2']=np.cos(4*np.pi*df['hour']/24)
df['dow_sin']=np.sin(2*np.pi*df['dayofweek']/7); df['dow_cos']=np.cos(2*np.pi*df['dayofweek']/7)
df['month_sin']=np.sin(2*np.pi*df['month']/12); df['month_cos']=np.cos(2*np.pi*df['month']/12)
df['doy_sin']=np.sin(2*np.pi*df['doy']/365); df['doy_cos']=np.cos(2*np.pi*df['doy']/365)
df['temp_squared']=df['temp']**2
for lag in [24,48,168,336]: df[f'lag_{lag}']=df[TARGET_COL].shift(lag)
df['roll_mean_24']=df[TARGET_COL].shift(24).rolling(24).mean()
df['roll_std_24']=df[TARGET_COL].shift(24).rolling(24).std()
df['roll_mean_168']=df[TARGET_COL].shift(24).rolling(168).mean()
df['workday_hour_sin']=df['workday']*df['hour_sin']; df['workday_hour_cos']=df['workday']*df['hour_cos']
df['lag_mean_same_type']=(df.groupby(['workday','hour'])[TARGET_COL]
    .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean()))
OPTIONAL=[c for c in ['wet','cloud','windspeed','daylength','day/night'] if c in df.columns]
FEATURES=['hour_sin','hour_cos','hour_sin_2','hour_cos_2','dow_sin','dow_cos',
    'month_sin','month_cos','doy_sin','doy_cos','workday','is_weekend','is_monday','is_friday',
    'temp','temp_squared','lag_24','lag_48','lag_168','lag_336',
    'roll_mean_24','roll_std_24','roll_mean_168',
    'workday_hour_sin','workday_hour_cos','lag_mean_same_type']+OPTIONAL
df_clean=df.dropna(subset=FEATURES+[TARGET_COL]).reset_index(drop=True)
print(f'  После dropna: {len(df_clean)} строк', flush=True)

all_windows=[]; n=len(df_clean); start=TRAIN_HOURS
while start+FORECAST_HOURS<=n:
    ts_s=start; ts_e=start+FORECAST_HOURS
    if (df_clean['workday'].iloc[ts_s:ts_e]==WORKDAY_FLAG).all():
        all_windows.append((ts_s-TRAIN_HOURS, ts_s, ts_e))
    start+=STEP_HOURS
if not all_windows: raise RuntimeError('Нет окон!')
if len(all_windows)<=MAX_WINDOWS: windows=all_windows
else:
    idx=np.linspace(0,len(all_windows)-1,MAX_WINDOWS,dtype=int)
    windows=[all_windows[i] for i in idx]
print(f'  Окон: {len(windows)}', flush=True)

best_score=[float('inf')]
def objective(trial):
    params=dict(num_leaves=suggest_from_space(trial,'num_leaves',SEARCH_SPACE['num_leaves']),
        n_estimators=suggest_from_space(trial,'n_estimators',SEARCH_SPACE['n_estimators']),
        learning_rate=suggest_from_space(trial,'learning_rate',SEARCH_SPACE['learning_rate']),
        min_child_samples=suggest_from_space(trial,'min_child_samples',SEARCH_SPACE['min_child_samples']),
        subsample=suggest_from_space(trial,'subsample',SEARCH_SPACE['subsample']),
        colsample_bytree=suggest_from_space(trial,'colsample_bytree',SEARCH_SPACE['colsample_bytree']),
        reg_alpha=suggest_from_space(trial,'reg_alpha',SEARCH_SPACE['reg_alpha']),
        reg_lambda=suggest_from_space(trial,'reg_lambda',SEARCH_SPACE['reg_lambda']),
        verbosity=-1,random_state=42,device='gpu',gpu_device_id=0,gpu_use_dp=False)
    mapes,r2s=[],[]
    for (tr_s,ts_s,ts_e) in windows:
        df_tr=df_clean.iloc[tr_s:ts_s]; df_tr=df_tr[df_tr['workday']==WORKDAY_FLAG]
        if len(df_tr)<24: continue
        df_ts=df_clean.iloc[ts_s:ts_e]
        m=LGBMRegressor(**params)
        with warnings.catch_warnings(): warnings.simplefilter('ignore'); m.fit(df_tr[FEATURES],df_tr[TARGET_COL].values)
        yp=m.predict(df_ts[FEATURES]); yt=df_ts[TARGET_COL].values
        mask=yt!=0
        if not mask.sum(): continue
        mapes.append(float(np.mean(np.abs((yt[mask]-yp[mask])/yt[mask]))*100))
        r2s.append(float(r2_score(yt,yp)))
    if not mapes: return float('inf')
    score=W_MAPE*float(np.mean(mapes))+W_R2*(1.0-float(np.min(r2s)))
    if score<best_score[0]: best_score[0]=score
    pbar.set_postfix({'best':f'{best_score[0]:.5f}'}); pbar.update(1)
    return score

print(f'\nOptuna: {N_TRIALS} триалов (workday, GPU)', flush=True)
done_flag=[False]
pbar=tqdm(total=N_TRIALS,desc='  Триалы',file=sys.stdout,dynamic_ncols=True,mininterval=0)
def _ticker():
    while not done_flag[0]: pbar.refresh(); time.sleep(1.0)
threading.Thread(target=_ticker,daemon=True).start()
study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective,n_trials=N_TRIALS,show_progress_bar=False)
done_flag[0]=True; pbar.close()
best_params=study.best_params; best_value=study.best_value
print('\n'+'='*60, flush=True)
for k,v in best_params.items(): print(f'  {k:25s} = {v}', flush=True)
print(f'  score = {best_value:.6f}', flush=True)

result={'specialist':SPECIALIST,'tune_start':str(TUNE_START.date()),
    'tune_end':str((TUNE_END-pd.Timedelta(days=1)).date()),
    'n_trials':N_TRIALS,'train_days':TRAIN_DAYS,'n_windows':len(windows),
    'best_score':best_value,'params':best_params}
json_path=os.path.join(OUTPUT_DIR,f'best_params_{SPECIALIST}.json')
with open(json_path,'w',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2)
print(f'✓ {json_path}', flush=True)
study.trials_dataframe().to_csv(os.path.join(OUTPUT_DIR,f'trials_{SPECIALIST}.csv'),index=False)

best_params_full={**best_params,'verbosity':-1,'random_state':42,'device':'gpu','gpu_device_id':0,'gpu_use_dp':False}
df_final=df_clean[df_clean['workday']==WORKDAY_FLAG]
fm=LGBMRegressor(**best_params_full)
with warnings.catch_warnings(): warnings.simplefilter('ignore'); fm.fit(df_final[FEATURES],df_final[TARGET_COL].values)
pkl_path=os.path.join(OUTPUT_DIR,f'best_model_{SPECIALIST}.pkl')
with open(pkl_path,'wb') as f:
    import pickle
    pickle.dump({'model':fm,'features':FEATURES,'params':best_params_full,'score':best_value,'specialist':SPECIALIST},f)
print(f'✓ {pkl_path}', flush=True)
print('\nГотово!', flush=True)
