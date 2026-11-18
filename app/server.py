"""Local-only web UI and asynchronous bridge to the existing stitching CLI."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import argparse
import json
import mimetypes
import re
import subprocess
import sys
import threading
import time
import uuid
import webbrowser

APP = Path(__file__).resolve().parent
PROJECT = APP.parent
EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
MAX_FILE = 512 * 1024**2
MAX_TOTAL = 4 * 1024**3
PREPROCESSORS = {'dehaze': 'dehaze-report.json', 'lowlight': 'enhancement-report.json'}


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.jobs = {}
        self.lock = threading.RLock()
        self.active = None

    def create(self, data):
        names = data.get('files', [])
        if not isinstance(names, list) or not 2 <= len(names) <= 100:
            raise ValueError('请选择 2～100 张待拼接图片。')
        if any(not isinstance(n, str) or len(n) > 240 or '/' in n or '\\' in n
               or Path(n).suffix.lower() not in EXTENSIONS for n in names):
            raise ValueError('图片文件名或格式不支持。')
        if len({n.casefold() for n in names}) != len(names):
            raise ValueError('存在重复图片文件名，请只导入同一级目录的图片。')
        mode = data.get('mode', 'auto')
        blend = data.get('blend', 'sharp')
        columns = data.get('columns', 4)
        if mode not in ('auto', 'grid') or blend not in ('sharp', 'feather'):
            raise ValueError('拼接模式或融合方式无效。')
        if type(columns) is not int or not 1 <= columns <= (len(names) if mode == 'grid' else 100):
            raise ValueError('网格列数必须在 1 和图片数量之间。')
        identity = uuid.uuid4().hex
        folder = self.root / identity
        (folder / 'inputs').mkdir(parents=True)
        job = dict(id=identity, status='uploading', files=names, uploaded=[], sizes={},
                   mode=mode, blend=blend, columns=columns, logs=[], error=None,
                   result=None, processed=None, preprocess_method=None, use_processed=False,
                   stage='upload', created=time.time(), folder=folder, process=None)
        with self.lock:
            self.jobs[identity] = job
        return job

    def public(self, job):
        return {k: job[k] for k in ('id', 'status', 'files', 'uploaded', 'mode', 'blend',
                                    'columns', 'logs', 'error', 'result', 'created', 'processed', 'stage', 'preprocess_method')}

    def start(self, job, options=None):
        with self.lock:
            if job['status'] not in ('uploading', 'ready') or len(job['uploaded']) != len(job['files']):
                raise ValueError('图片尚未全部上传，或任务已经启动。')
            if self.active is not None:
                raise ValueError('已有任务正在拼接，请稍后重试。')
            options = options or {}
            mode, blend, columns = (options.get(k, job[k]) for k in ('mode', 'blend', 'columns'))
            if mode not in ('auto', 'grid') or blend not in ('sharp', 'feather'):
                raise ValueError('拼接模式或融合方式无效。')
            if type(columns) is not int or not 1 <= columns <= (len(job['files']) if mode == 'grid' else 100):
                raise ValueError('网格列数必须在 1 和图片数量之间。')
            use_processed = options.get('use_processed', job['processed'] is not None)
            if type(use_processed) is not bool:
                raise ValueError('前处理输入选项无效。')
            if use_processed and job['processed'] is None:
                raise ValueError('前处理结果尚未准备好，请先应用前处理。')
            job.update(mode=mode, blend=blend, columns=columns, use_processed=use_processed, stage='stitch', error=None)
            self.active = job['id']
            job['status'] = 'running'
        threading.Thread(target=self.run, args=(job,), daemon=True).start()

    def input_path(self, job, index):
        return job['folder'] / 'inputs' / (f'{index:04d}' + Path(job['files'][index]).suffix.lower())

    def processed_path(self, job, index):
        return job['folder'] / 'processed' / f'{index:04d}.png'

    def registration_path(self, job, index):
        return job['folder'] / 'processed' / 'registration' / f'{index:04d}.png'

    def preprocess(self, job, method):
        if not isinstance(method, str) or method not in PREPROCESSORS:
            raise ValueError('不支持的前处理方式。')
        with self.lock:
            if job['status'] not in ('uploading', 'ready', 'preprocess_failed') or len(job['uploaded']) != len(job['files']):
                raise ValueError('请先上传全部图片，再应用前处理。')
            if self.active is not None:
                raise ValueError('已有前处理或拼接任务正在运行，请稍后重试。')
            job.update(status='preprocessing', stage='preprocess', error=None, processed=None,
                       preprocess_method=method, logs=[])
            self.active = job['id']
        threading.Thread(target=self.run_preprocess, args=(job,), daemon=True).start()

    def execute(self, job, args, log_name):
        with (job['folder'] / log_name).open('w', encoding='utf-8') as log:
            process = subprocess.Popen(args, cwd=PROJECT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                       errors='replace', creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            job['process'] = process
            with process.stdout as lines:
                for line in lines:
                    line = line.rstrip()
                    log.write(line + '\n'); log.flush()
                    with self.lock:
                        job['logs'] = (job['logs'] + [line])[-400:]
            code = process.wait()
        if code != 0:
            errors = [line[7:] for line in job['logs'] if line.startswith('Error: ')]
            raise RuntimeError(errors[-1] if errors else f'处理进程异常退出（代码 {code}），请查看运行日志。')

    def persist(self, job):
        with self.lock:
            if self.active == job['id']:
                self.active = None
            job['process'] = None
            (job['folder'] / 'status.json').write_text(json.dumps(self.public(job), ensure_ascii=False, indent=2), encoding='utf-8')

    def run_preprocess(self, job):
        method = job['preprocess_method']
        args = [sys.executable, '-u', str(APP / 'preprocess_worker.py'), '--method', method, '--input', str(job['folder'] / 'inputs'),
                '--output', str(job['folder'] / 'processed'), '--pattern', '*', '--overwrite']
        try:
            self.execute(job, args, 'preprocess.log')
            report = json.loads((job['folder'] / 'processed' / PREPROCESSORS[method]).read_text(encoding='utf-8'))
            if not report['complete'] or len(report['images']) != len(job['files']):
                raise ValueError('前处理输出数量不完整，请重新应用。')
            entries = []
            for i, name in enumerate(job['files']):
                if not self.processed_path(job, i).is_file():
                    raise ValueError(f'缺少前处理输出：{name}')
                if method == 'lowlight' and not self.registration_path(job, i).is_file():
                    raise ValueError(f'缺少暗光增强的配准辅助图：{name}，请重新应用。')
                entries.append(dict(index=i, name=name, size=report['images'][i]['size'],
                    preview=f"/api/jobs/{job['id']}/processed/{i}/preview",
                    full=f"/api/jobs/{job['id']}/processed/{i}"))
            with self.lock:
                job['processed'] = dict(method=method, images=entries, report=report)
        except Exception as exc:
            with self.lock:
                job.update(error=str(exc), processed=None)
        finally:
            with self.lock:
                job['status'] = 'ready' if job['processed'] is not None else 'preprocess_failed'
                self.persist(job)

    def run(self, job):
        folder = job['folder']
        args = [sys.executable, '-u', str(APP / 'worker.py'), 'stitch', '--images']
        inputs = self.processed_path if job['use_processed'] else self.input_path
        args += [str(inputs(job, i)) for i in range(len(job['files']))]
        if job['use_processed'] and job['processed']['method'] == 'lowlight' and job['mode'] == 'auto':
            args += ['--matching-images'] + [str(self.registration_path(job, i)) for i in range(len(job['files']))]
        args += ['-o', str(folder / 'result.png'), '--blend', job['blend']]
        if job['mode'] == 'grid':
            args += ['--layout-grid', str(job['columns'])]
        try:
            self.execute(job, args, 'run.log')
            report = json.loads((folder / 'result.report.json').read_text(encoding='utf-8'))
            w, h = report['canvas_size']
            with self.lock:
                job['result'] = dict(width=w, height=h,
                    preview=f"/api/jobs/{job['id']}/preview" if (folder / 'preview.jpg').exists() else f"/api/jobs/{job['id']}/result",
                    download=f"/api/jobs/{job['id']}/result", output=str(folder / 'result.png'), report=report)
                job['status'] = 'success'
                self.active = None
        except Exception as exc:
            with self.lock:
                job['status'] = 'failed'; job['error'] = str(exc)
                self.active = None
        finally:
            self.persist(job)


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, root):
        self.store = Store(root)
        super().__init__(address, Handler)

    def server_close(self):
        for job in self.store.jobs.values():
            process = job.get('process')
            if process is not None and process.poll() is None:
                process.terminate()
        super().server_close()


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *_):
        pass

    def json(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers(); self.wfile.write(body)

    def file(self, path, download=False):
        if not path.is_file():
            self.json({'error': '文件尚未生成或不存在。'}, 404); return
        self.send_response(200)
        self.send_header('Content-Type', mimetypes.guess_type(str(path))[0] or 'application/octet-stream')
        self.send_header('Content-Length', str(path.stat().st_size))
        self.send_header('X-Content-Type-Options', 'nosniff')
        if path.parent == APP / 'static':
            self.send_header('Cache-Control', 'no-store')
        if download:
            self.send_header('Content-Disposition', 'attachment; filename="panorama.png"')
        self.end_headers()
        with path.open('rb') as stream:
            while chunk := stream.read(1024 * 1024):
                self.wfile.write(chunk)

    def request_json(self):
        length = int(self.headers.get('Content-Length', '0'))
        if not 0 < length <= 100_000:
            raise ValueError('请求内容过大或为空。')
        data = json.loads(self.rfile.read(length))
        if not isinstance(data, dict):
            raise ValueError('请求格式无效。')
        return data

    def route(self):
        origin = self.headers.get('Origin')
        host = self.headers.get('Host', '')
        allowed = {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}
        if host not in allowed or (origin and urlsplit(origin).netloc != host):
            # Drain a small JSON request before closing. On Windows an unread
            # body can reset the connection before the client receives the 403.
            length = self.headers.get('Content-Length', '0')
            if length.isdecimal() and 0 < int(length) <= 100_000:
                self.rfile.read(int(length))
            self.close_connection = True
            self.json({'error': '仅允许本机同源访问。'}, 403); return
        path = urlsplit(self.path).path
        if self.command == 'GET' and path == '/api/health':
            self.json({'ok': True}); return
        if self.command == 'POST' and path == '/api/jobs':
            self.json(self.server.store.public(self.server.store.create(self.request_json())), 201); return
        match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})(?:/(run|preprocess|result|preview|files/([0-9]+)))?', path)
        processed_match = re.fullmatch(r'/api/jobs/([a-f0-9]{32})/processed/([0-9]+)(/preview)?', path)
        if processed_match and self.command == 'GET':
            identity, index, thumbnail = processed_match.groups()
            job = self.server.store.jobs.get(identity)
            index = int(index)
            if job is None or job['processed'] is None or not 0 <= index < len(job['files']):
                self.json({'error': '前处理图片尚未准备好或不存在。'}, 404); return
            target = job['folder'] / 'processed' / 'previews' / f'{index:04d}.jpg' if thumbnail else self.server.store.processed_path(job, index)
            self.file(target); return
        if match:
            identity, action, index = match.groups()
            job = self.server.store.jobs.get(identity)
            if job is None:
                self.json({'error': '任务不存在，服务重启后请重新导入。'}, 404); return
            if self.command == 'GET' and action is None:
                with self.server.store.lock:
                    self.json(self.server.store.public(job))
                return
            if self.command == 'GET' and action in ('result', 'preview'):
                self.file(job['folder'] / ('result.png' if action == 'result' else 'preview.jpg'), action == 'result'); return
            if self.command == 'POST' and action == 'run':
                self.server.store.start(job, self.request_json())
                self.json(self.server.store.public(job)); return
            if self.command == 'POST' and action == 'preprocess':
                data = self.request_json()
                self.server.store.preprocess(job, data.get('method'))
                self.json(self.server.store.public(job)); return
            if self.command == 'PUT' and index is not None:
                index = int(index)
                if not 0 <= index < len(job['files']) or job['status'] != 'uploading':
                    raise ValueError('上传序号或任务状态无效。')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= MAX_FILE or sum(job['sizes'].values()) - job['sizes'].get(index, 0) + length > MAX_TOTAL:
                    raise ValueError('单张图片上限 512 MB，每个任务上限 4 GB。')
                remaining = length
                with self.server.store.input_path(job, index).open('wb') as stream:
                    while remaining:
                        chunk = self.rfile.read(min(remaining, 1024 * 1024))
                        if not chunk:
                            raise ValueError('图片上传中断，请重新导入。')
                        stream.write(chunk); remaining -= len(chunk)
                with self.server.store.lock:
                    job['sizes'][index] = length
                    job['uploaded'] = sorted(job['sizes'])
                self.json({'uploaded': index}); return
        if self.command == 'GET' and path in ('/', '/index.html', '/style.css', '/app.js'):
            self.file(APP / 'static' / ('index.html' if path == '/' else path[1:])); return
        self.json({'error': '接口不存在。'}, 404)

    def handle_request(self):
        try:
            self.route()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            self.close_connection = True
            self.json({'error': str(exc)}, 400)
        except Exception as exc:
            self.close_connection = True
            self.json({'error': f'服务处理失败：{exc}'}, 500)

    do_GET = do_POST = do_PUT = handle_request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--open', action='store_true')
    args = parser.parse_args()
    with Server(('127.0.0.1', args.port), PROJECT / 'output' / 'web') as server:
        url = f'http://127.0.0.1:{server.server_port}'
        print(f'Local stitching UI: {url}', flush=True)
        print('Press Ctrl+C to stop.', flush=True)
        if args.open:
            threading.Timer(.5, lambda: webbrowser.open(url)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
