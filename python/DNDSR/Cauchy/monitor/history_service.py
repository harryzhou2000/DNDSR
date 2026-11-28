"""Read-only companion HTTP service; stdlib only, independent of native bindings.

Implements the same paged numeric CSV protocol as Cauchy's embedded server.
Only manifest-declared histories are accessible, never a caller-supplied path.
"""
from __future__ import annotations
import argparse
import csv
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit


class HistoryIndex:
    def __init__(self):
        self.lock = threading.RLock()
        self.series = {}

    @staticmethod
    def paths(base):
        paths = [(0, base)] if base.is_file() else []
        for p in base.parent.glob(base.stem + '_*' + base.suffix):
            suffix = p.stem[len(base.stem) + 1:]
            if suffix.isdigit():
                paths.append((int(suffix), p))
        return [p for _, p in sorted(paths)]

    @staticmethod
    def tail(path, end):
        with path.open('rb') as f:
            f.seek(max(0, end - 64))
            return f.read(min(64, end))

    def query(self, base, args):
        with self.lock:
            return self._query(Path(base), args)

    def _query(self, base, args):
        limit = int(args.get('limit', 2000))
        if not 1 <= limit <= 10000:
            raise ValueError('limit must be 1..10000')
        kind, axis = args.get('kind', 'all'), args.get('axis', 'iteration')
        if kind not in ('all', 'nonlinear', 'stage', 'step') or axis not in ('iteration', 'time', 'step', 'wall'):
            raise ValueError('invalid kind or axis')
        lo, hi = float(args.get('start', '-inf')
                       ), float(args.get('end', 'inf'))
        if ('start' in args and not math.isfinite(lo)) or ('end' in args and not math.isfinite(hi)) or lo > hi:
            raise ValueError('invalid interval')
        cache = self.series.setdefault(
            str(base), {'generation': secrets.token_hex(12), 'segments': []})
        paths = self.paths(base)
        old = cache['segments']
        reset_index = len(paths) < len(old)
        for i, s in enumerate(old):
            if reset_index:
                break
            stat = paths[i].stat()
            if (paths[i] != s['path'] or stat.st_size < s['size'] or self.tail(paths[i], s['end']) != s['tail']
                    or (stat.st_size == s['size'] and stat.st_mtime_ns != s['stamp'])
                    or (i + 1 < len(old) and stat.st_size != s['size'])):
                reset_index = True
        if reset_index:
            cache.update(generation=secrets.token_hex(12), segments=[])
        for i, path in enumerate(paths):
            if i == len(cache['segments']):
                cache['segments'].append(
                    dict(path=path, end=0, rows=0, offsets=[], header=[]))
            s = cache['segments'][i]
            with path.open('rb') as f:
                f.seek(s['end'])
                while True:
                    start = f.tell()
                    line = f.readline()
                    if not line.endswith(b'\n'):
                        break
                    if not s['header']:
                        s['header'] = next(csv.reader([line.decode()]))
                    else:
                        if s['rows'] % 256 == 0:
                            s['offsets'].append(start)
                        s['rows'] += 1
                    s['end'] = f.tell()
            stat = path.stat()
            s.update(size=stat.st_size, stamp=stat.st_mtime_ns,
                     tail=self.tail(path, s['end']))
        header = next((s['header']
                      for s in cache['segments'] if s['header']), [])
        if any(s['header'] and s['header'] != header for s in cache['segments']):
            raise ValueError('history segment schema changed')
        total = sum(s['rows'] for s in cache['segments'])
        position, reset = 0, False
        if args.get('cursor'):
            generation, raw_position = args['cursor'].split(':')
            position = int(raw_position)
            if position < 0:
                raise ValueError('invalid cursor')
            reset = generation != cache['generation'] or position > total
            if reset:
                position = 0
        columns = args['columns'].split(',') if args.get('columns') else header
        if header and any(c not in header for c in columns):
            raise ValueError('unknown column')
        rows, indices, base_row = [], [], 0
        for s in cache['segments']:
            if position >= base_row + s['rows']:
                base_row += s['rows']
                continue
            local = position - base_row
            with s['path'].open('rb') as f:
                f.seek(s['offsets'][local // 256])
                for _ in range(local % 256):
                    f.readline()
                while local < s['rows'] and len(rows) < limit:
                    values = next(csv.reader([f.readline().decode()]))
                    if len(values) != len(header):
                        raise ValueError('malformed complete CSV row')
                    row = {}
                    for k, v in zip(header, values):
                        try:
                            n = float(v)
                            row[k] = n if math.isfinite(n) else None
                        except ValueError:
                            row[k] = None
                    ordinal = base_row + local
                    local += 1
                    position = ordinal + 1
                    if kind != 'all' and row.get('status') != {'nonlinear': 0, 'stage': 1, 'step': 2}[kind]:
                        continue
                    if (math.isfinite(lo) or math.isfinite(hi)) and (row.get(axis) is None or not lo <= row[axis] <= hi):
                        continue
                    rows.append({k: row[k] for k in columns})
                    indices.append(ordinal)
            if len(rows) == limit:
                break
            base_row += s['rows']
        return dict(schema='dndsr.cauchy.history-query.v1', source='csv', header=columns, rows=rows,
                    indices=indices, totalRows=total, returnedRows=len(rows), generation=cache['generation'],
                    reset=reset, hasMore=position < total, nextCursor=f"{cache['generation']}:{position}")


class Companion:
    def __init__(self, manifest, token):
        self.manifest_path = Path(manifest).resolve()
        self.token = token
        self.index = HistoryIndex()

    def manifest(self):
        return json.loads(self.manifest_path.read_text())

    def resolve(self, name):
        p = Path(name)
        if p.is_absolute():
            return p
        # Relocated runs keep their output files adjacent to the manifest.
        adjacent = self.manifest_path.parent / p.name
        return adjacent if adjacent.exists() else p.resolve()

    def get(self, route, args):
        manifest = self.manifest()
        if route == '/api/v1/manifest':
            return manifest
        if route == '/api/v1/config':
            return {'configuration': json.loads(self.resolve(manifest['resolvedConfig']).read_text()), 'mutableSchema': {}, 'updates': []}
        if route == '/api/v1/status':
            return dict(runId='archive-' + self.manifest_path.stem, state='archived', prefix=str(self.manifest_path),
                        runManifest=str(self.manifest_path), pid=os.getpid(), readOnly=True,
                        domains=len(manifest.get('domains', [])), updates=[])
        if route == '/api/v1/boundaries':
            return {'latest': {}}
        if route != '/api/v1/history/query':
            raise KeyError('unknown endpoint')
        domain = int(args.get('domain', 0))
        for d in manifest['domains']:
            if d['index'] != domain:
                continue
            if 'boundary' not in args:
                return self.index.query(self.resolve(d['history']), args)
            for b in d.get('boundaryIntegrations', []):
                if b['canonicalName'] == args['boundary']:
                    return self.index.query(self.resolve(b['csv']), args)
        raise KeyError('history not found')


def serve(companion, host, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass  # never log credentials or query strings

        def do_GET(self):
            supplied = self.headers.get(
                'Authorization', '').removeprefix('Bearer ')
            if not hmac.compare_digest(supplied, companion.token):
                self.reply(401, {'error': 'unauthorized'})
                return
            request = urlsplit(self.path)
            try:
                self.reply(200, companion.get(request.path, {
                           k: v[-1] for k, v in parse_qs(request.query).items()}))
            except KeyError as e:
                self.reply(404, {'error': str(e)})
            except (ValueError, TypeError) as e:
                self.reply(400, {'error': str(e)})
            except OSError as e:
                self.reply(503, {'error': str(e)})

        def reply(self, status, value):
            body = json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=53000)
    parser.add_argument('--token-file', required=True,
                        help='Private bearer-token file; created with mode 0600 if absent')
    args = parser.parse_args()
    token_file = Path(args.token_file)
    if not token_file.exists():
        with os.fdopen(os.open(token_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600), 'w') as f:
            f.write(secrets.token_hex(32) + '\n')
    token = token_file.read_text().strip()
    if not token:
        parser.error('empty token')
    server = serve(Companion(args.manifest, token), args.host, args.port)
    print(
        f'History companion listening on http://{args.host}:{server.server_port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
