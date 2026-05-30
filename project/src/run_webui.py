"""
project/src/run_webui.py  —  канонический entrypoint.

Запуск:
    python project/src/run_webui.py
    # или через root-proxy:
    python run_webui.py
    # или через Docker:
    docker compose up --build
"""
import os
import sys

# Добавляем project/src/web_interface/ в sys.path,
# чтобы Flask мог найти templates/ и импортировать app.py
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR  = os.path.join(THIS_DIR, 'web_interface')
sys.path.insert(0, WEB_DIR)
sys.path.insert(0, THIS_DIR)

from app import app
import threading
import time
import webbrowser

host = os.environ.get('FLASK_HOST', '127.0.0.1')
port = int(os.environ.get('FLASK_PORT', 5000))

print(f'Интерфейс запущен: http://{host}:{port}', flush=True)

if host in ('127.0.0.1', 'localhost'):
    threading.Thread(
        target=lambda: (time.sleep(1.5), webbrowser.open(f'http://127.0.0.1:{port}')),
        daemon=True
    ).start()

app.run(host=host, port=port, debug=False, threaded=True)
