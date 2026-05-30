"""
run_specialists_gpu.py  (project/src/avto_test_py/)
====================================================
Запускает поочерёдно tune_workday_gpu.py и tune_nonworkday_gpu.py.
Читает конфиги из project/configs/s_config.json.

Запуск (из корня репозитория):
    python project/src/avto_test_py/run_specialists_gpu.py
"""
import os
import sys
import subprocess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

scripts = [
    os.path.join(THIS_DIR, 'tune_workday_gpu.py'),
    os.path.join(THIS_DIR, 'tune_nonworkday_gpu.py'),
]

for script in scripts:
    print(f'\n{"="*60}', flush=True)
    print(f'Запуск: {script}', flush=True)
    print('='*60, flush=True)
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUTF8'] = '1'
    result = subprocess.run(
        [sys.executable, '-X', 'utf8', '-u', script],
        env=env,
    )
    if result.returncode != 0:
        print(f'\n✗ Скрипт завершился с кодом {result.returncode}: {script}', flush=True)
        sys.exit(result.returncode)
    print(f'✓ Готово: {script}', flush=True)

print('\n✓ Все специалисты обучены!', flush=True)
