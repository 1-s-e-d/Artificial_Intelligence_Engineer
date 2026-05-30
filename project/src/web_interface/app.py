import os
import sys
import re
import json
import time
import subprocess
import threading
import webbrowser

from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import pandas as pd

# ---------- пути -----------
# project/src/web_interface/app.py
#   BASE_DIR  = .../project/src/web_interface
#   SRC_DIR   = .../project/src
#   ROOT_DIR  = корень репозитория (3 уровня вверх)
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.dirname(BASE_DIR)
ROOT_DIR   = os.path.dirname(os.path.dirname(SRC_DIR))  # repo root

CONFIG_PATH        = os.path.join(ROOT_DIR, 'project', 'configs', 's_config.json')
DEVICE_CONFIG_PATH = os.path.join(ROOT_DIR, 'project', 'configs', 'device_config.json')
DATA_PATH          = os.path.join(ROOT_DIR, 'data', 'dataset.xlsx')

TUNE_SCRIPT     = os.path.join(SRC_DIR, 'avto_test_py', 'tune_allday_gpu.py')
FORECAST_SCRIPT = os.path.join(ROOT_DIR, 'project', 'tests', 'test-34.py')
BENCH_SCRIPT    = os.path.join(SRC_DIR, 'avto_test_py', 'benchmark_device.py')

ANSI_ESCAPE  = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
SEPARATOR_RE = re.compile(r'^[=\-*#_]{3,}$')
TQDM_STEP_RE = re.compile(r'\b(\d+)/\d+')

# Шаблоны и статика лежат рядом с app.py
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'templates'),
    static_folder=os.path.join(BASE_DIR, 'static'),
)

# ---------- глобальное состояние ----------
task_lock = threading.Lock()

task_status = {
    'type': None,
    'running': False,
    'proc': None,
    'log_buf': [],
    'log_cond': threading.Condition(),
    'stream_done': threading.Event(),
    'task_id': 0,
}


# ---------- вспомогательные ----------
def load_config():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_config(cfg):
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


def load_device_config():
    if os.path.exists(DEVICE_CONFIG_PATH):
        with open(DEVICE_CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'device': 'cpu', 'n_jobs': 1, 'num_threads': -1}


def save_device_config(dcfg):
    os.makedirs(os.path.dirname(DEVICE_CONFIG_PATH), exist_ok=True)
    with open(DEVICE_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(dcfg, f, indent=2)


def get_data_summary():
    df = pd.read_excel(DATA_PATH)
    date_col = 'dateByOurs'
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    if 'workday' not in df.columns:
        df['workday'] = (df[date_col].dt.dayofweek < 5).astype(int)
    df = df[pd.to_numeric(df['workday'], errors='coerce').notna()].copy()
    df['workday'] = df['workday'].astype(int)
    df['_date'] = df[date_col].dt.normalize()
    dates_df = (
        df[['_date', 'workday']]
        .drop_duplicates(subset=['_date'])
        .sort_values('_date')
    )
    data_days = {}
    for _, row in dates_df.iterrows():
        d = row['_date'].strftime('%Y-%m-%d')
        data_days[d] = int(row['workday'])

    tune_days = set()
    forecast_days = set()
    train_days_set = set()

    try:
        cfg = load_config()
        tune_start    = pd.Timestamp(cfg['tune']['tune_start'])
        tune_end      = pd.Timestamp(cfg['tune']['tune_end'])
        n_tune_window = int(cfg['model'].get('train_days', 90))
        tune_data_end = tune_end + pd.Timedelta(days=n_tune_window)
        for d in pd.date_range(tune_start, tune_data_end, freq='D'):
            tune_days.add(d.strftime('%Y-%m-%d'))
    except Exception:
        pass

    try:
        cfg = load_config()
        fc = cfg['forecast']
        train_start = pd.Timestamp(fc['train_start'])
        train_end   = pd.Timestamp(fc['train_end'])
        for d in pd.date_range(train_start, train_end, freq='D'):
            train_days_set.add(d.strftime('%Y-%m-%d'))
        fore_start = pd.Timestamp(fc['forecast_start'])
        fore_end   = pd.Timestamp(fc['forecast_end'])
        for d in pd.date_range(fore_start, fore_end, freq='D'):
            forecast_days.add(d.strftime('%Y-%m-%d'))
    except Exception:
        pass

    all_dates = set(data_days.keys()) | tune_days | forecast_days | train_days_set
    events = []
    for d in sorted(all_dates):
        types = []
        if d in data_days:
            types.append('data')
        if d in tune_days:
            types.append('tune')
        if d in train_days_set:
            types.append('train')
        if d in forecast_days:
            if d in data_days:
                types.append('forecast')
            else:
                types.append('forecast_only')
        events.append({'start': d, 'types': types, 'workday': data_days.get(d, -1)})
    return events


# ---------- чтение stdout ----------
def _buf_put(item):
    cond = task_status['log_cond']
    with cond:
        task_status['log_buf'].append(item)
        cond.notify_all()


def read_stream(proc_stdout_fd):
    buf = b''
    cr_pending = ''
    cr_step = None

    def _decode(raw):
        if isinstance(raw, (bytes, bytearray)):
            text = ANSI_ESCAPE.sub('', raw.decode('utf-8', errors='replace'))
        else:
            text = ANSI_ESCAPE.sub('', raw)
        return text.rstrip('\r\n').strip()

    def _step(text):
        m = TQDM_STEP_RE.search(text)
        return int(m.group(1)) if m else None

    def _put(msg_type, text):
        if not text or SEPARATOR_RE.match(text):
            return
        _buf_put({'type': msg_type, 'text': text})

    while True:
        try:
            ch = os.read(proc_stdout_fd, 1)
        except OSError:
            ch = b''

        if not ch:
            if cr_pending:
                _put('log', cr_pending)
            elif buf:
                _put('log', _decode(buf))
            break

        if ch == b'\n':
            line_text = _decode(buf)
            buf = b''
            if cr_pending:
                cr_pending = ''
                cr_step = None
            else:
                _put('log', line_text)

        elif ch == b'\r':
            if not buf:
                continue
            new_text = _decode(buf)
            new_step = _step(new_text)
            buf = b''

            if cr_pending:
                if new_step is not None and cr_step is not None and new_step != cr_step:
                    _put('log', cr_pending)

            cr_pending = new_text
            cr_step = new_step
            if new_text and not SEPARATOR_RE.match(new_text):
                _buf_put({'type': 'progress', 'text': new_text})
        else:
            buf += ch

    if cr_pending:
        _put('log', cr_pending)
    elif buf:
        _put('log', _decode(buf))
    _buf_put({'type': '_eof', 'text': ''})


def run_script_task(script_path, args, task_name):
    with task_lock:
        if task_status['running']:
            return
        task_status['type'] = task_name
        task_status['running'] = True
        task_status['proc'] = None
        task_status['task_id'] += 1
        with task_status['log_cond']:
            task_status['log_buf'] = []
        task_status['stream_done'].clear()

    try:
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONUTF8'] = '1'
        cmd = [sys.executable, '-X', 'utf8', '-u', script_path] + args
        proc = subprocess.Popen(
            cmd, cwd=ROOT_DIR,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=env, bufsize=0
        )
        with task_lock:
            task_status['proc'] = proc

        stdout_fd = proc.stdout.fileno()
        if sys.platform == 'win32':
            import msvcrt
            msvcrt.setmode(stdout_fd, os.O_BINARY)

        t = threading.Thread(target=read_stream, args=(stdout_fd,), daemon=True)
        t.start()

        proc.wait()
        t.join(timeout=5)

        if proc.returncode not in (0, None, -15, -9):
            _buf_put({'type': 'log', 'text': f'Process exited with code {proc.returncode}'})
        elif proc.returncode == 0:
            _buf_put({'type': 'log', 'text': 'Task completed successfully.'})
        else:
            _buf_put({'type': 'log', 'text': 'Task was stopped.'})

    except Exception as e:
        _buf_put({'type': 'log', 'text': f'Exception: {e}'})
    finally:
        with task_lock:
            task_status['running'] = False
            task_status['proc'] = None
        task_status['stream_done'].set()
        with task_status['log_cond']:
            task_status['log_cond'].notify_all()


# ---------- маршруты ----------
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/calendar_events')
def calendar_events():
    return jsonify(get_data_summary())


@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    if request.method == 'GET':
        return jsonify(load_config())
    new_cfg = request.get_json()
    if new_cfg:
        save_config(new_cfg)
        return jsonify({'status': 'ok'})
    return jsonify({'status': 'error', 'message': 'No data'}), 400


@app.route('/api/device_config', methods=['GET', 'POST'])
def api_device_config():
    if request.method == 'GET':
        return jsonify(load_device_config())
    dcfg = request.get_json()
    save_device_config(dcfg)
    return jsonify({'status': 'ok'})


@app.route('/api/run_tune', methods=['POST'])
def run_tune():
    if task_status['running']:
        return jsonify({'status': 'busy'}), 409
    threading.Thread(target=run_script_task, args=(TUNE_SCRIPT, [], 'tune'), daemon=True).start()
    time.sleep(0.05)
    return jsonify({'status': 'started', 'task_id': task_status['task_id']})


@app.route('/api/run_forecast', methods=['POST'])
def run_forecast():
    if task_status['running']:
        return jsonify({'status': 'busy'}), 409
    threading.Thread(target=run_script_task, args=(FORECAST_SCRIPT, [], 'forecast'), daemon=True).start()
    time.sleep(0.05)
    return jsonify({'status': 'started', 'task_id': task_status['task_id']})


@app.route('/api/run_benchmark', methods=['POST'])
def run_benchmark():
    if task_status['running']:
        return jsonify({'status': 'busy'}), 409
    threading.Thread(target=run_script_task, args=(BENCH_SCRIPT, [], 'benchmark'), daemon=True).start()
    time.sleep(0.05)
    return jsonify({'status': 'started', 'task_id': task_status['task_id']})


@app.route('/api/stop', methods=['POST'])
def api_stop():
    with task_lock:
        proc = task_status.get('proc')
        running = task_status.get('running', False)
    if not running or proc is None:
        return jsonify({'status': 'not_running'})
    try:
        proc.terminate()
        time.sleep(0.4)
        if proc.poll() is None:
            proc.kill()
        return jsonify({'status': 'stopped'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/shutdown', methods=['POST'])
def api_shutdown():
    with task_lock:
        proc = task_status.get('proc')
        running = task_status.get('running', False)
    if running and proc is not None:
        try:
            proc.terminate()
            time.sleep(0.5)
            if proc.poll() is None:
                proc.kill()
        except Exception:
            pass

    def _exit():
        time.sleep(0.5)
        os._exit(0)

    threading.Thread(target=_exit, daemon=True).start()
    return jsonify({'status': 'shutting_down'})


@app.route('/api/progress')
def progress():
    try:
        client_tid = int(request.args.get('tid', -1))
    except (ValueError, TypeError):
        client_tid = -1

    cond     = task_status['log_cond']
    done_evt = task_status['stream_done']

    def generate():
        cur_idx = 0

        while True:
            with cond:
                if client_tid != -1 and task_status['task_id'] != client_tid:
                    return

                buf = task_status['log_buf']
                if cur_idx >= len(buf):
                    if not task_status['running'] and done_evt.is_set():
                        break
                    cond.wait(timeout=0.5)
                    continue

                items = buf[cur_idx:]
                cur_idx += len(items)

            for item in items:
                if item['type'] == '_eof':
                    done_evt.wait(timeout=6)
                    done_payload = {'task': task_status.get('type', '')}
                    if os.path.exists(DEVICE_CONFIG_PATH):
                        try:
                            with open(DEVICE_CONFIG_PATH, 'r', encoding='utf-8') as f:
                                dcfg = json.load(f)
                            done_payload['device'] = dcfg.get('device', 'cpu')
                            times = dcfg.get('benchmark_times_sec', {})
                            done_payload['cpu_sec'] = round(times.get('cpu', 0), 2)
                            done_payload['gpu_sec'] = round(times.get('gpu', 0), 2)
                        except Exception:
                            pass
                    yield f"event: done\ndata: {json.dumps(done_payload)}\n\n"
                    return
                elif item['type'] == 'progress':
                    yield f"event: progress\ndata: {item['text']}\n\n"
                elif item['type'] == 'log':
                    yield f"data: {item['text']}\n\n"

            yield ': keepalive\n\n'

    headers = {
        'X-Accel-Buffering': 'no',
        'Cache-Control': 'no-cache',
        'Content-Type': 'text/event-stream',
    }
    return Response(stream_with_context(generate()), headers=headers)


if __name__ == '__main__':
    host = '127.0.0.1'
    port = 5000
    print(f'Интерфейс запущен: http://{host}:{port}')
    threading.Thread(
        target=lambda: (time.sleep(1.5), webbrowser.open(f'http://{host}:{port}')),
        daemon=True
    ).start()
    app.run(host=host, port=port, debug=False, threaded=True)
