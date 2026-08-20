"""Tiny static file server with HTTP Range support (206), for local showcase."""
import os
import posixpath
import sys
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = sys.argv[1] if len(sys.argv) > 1 else '.'


class RangeHandler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            path = os.path.join(path, 'index.html')
        if not os.path.isfile(path):
            self.send_error(404, 'Not found')
            return None
        ctype = self.guess_type(path)
        size = os.path.getsize(path)
        f = open(path, 'rb')
        rng = self.headers.get('Range')
        if rng:
            try:
                start, end = rng.replace('bytes=', '').split('-', 1)
                start = int(start)
                end = int(end) if end else size - 1
            except Exception:
                start, end = 0, size - 1
            if start >= size:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.end_headers()
                f.close()
                return None
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            f.seek(start)
            f = _Slice(f, start, end - start + 1)
        else:
            self.send_response(200)
        self.send_header('Content-type', ctype)
        self.send_header('Content-Length', str(getattr(f, '_len', size)))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('Last-Modified',
                         self.date_time_string(os.path.getmtime(path)))
        self.end_headers()
        return f


class _Slice:
    def __init__(self, f, start, length):
        self.f = f
        self._len = length
        self._left = length

    def read(self, n=-1):
        if n < 0 or n > self._left:
            n = self._left
        data = self.f.read(n)
        self._left -= len(data)
        return data

    def close(self):
        self.f.close()

    def __len__(self):
        return self._len


if __name__ == '__main__':
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8899
    srv = ThreadingHTTPServer(('0.0.0.0', port), RangeHandler)
    print(f'range server on 0.0.0.0:{port} serving {ROOT}', flush=True)
    srv.serve_forever()
