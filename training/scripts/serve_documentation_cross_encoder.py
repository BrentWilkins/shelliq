#!/usr/bin/env python3
"""Serve the passing documentation cross-encoder over loopback chat completions."""

from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shelliq_training.documentation_cross_encoder import DocumentationCrossEncoderRuntime

CHAT_PATH = '/v1/chat/completions'
MAX_REQUEST_BYTES = 1024 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--documentation-index', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--actions', type=Path, default=Path('../target/debug/semantic-actions'))
    parser.add_argument('--host', default='127.0.0.1', choices=('127.0.0.1', '::1'))
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    return parser.parse_args()


def handler(runtime: DocumentationCrossEncoderRuntime) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            if self.path != CHAT_PATH:
                self._json(404, {'error': 'not found'})
                return
            try:
                length = int(self.headers.get('Content-Length', ''))
                if not 1 <= length <= MAX_REQUEST_BYTES:
                    raise ValueError('request body is empty or too large')
                request = json.loads(self.rfile.read(length))
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
                self._json(400, {'error': str(error)})
                return
            try:
                response = runtime.chat_completion(request)
            except Exception as error:
                self._json(422, {'error': str(error)})
                return
            self._json(200, response)

        def _json(self, status: int, value: object) -> None:
            body = json.dumps(value, separators=(',', ':')).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, message: str, *args: object) -> None:
            print(f'documentation-ranker: {message % args}', file=sys.stderr)

    return Handler


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit('--port must be from 1 through 65535')
    runtime = DocumentationCrossEncoderRuntime(
        args.documentation_index,
        args.checkpoint,
        args.actions,
        args.device,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler(runtime))
    print(f'Documentation ranker listening on http://{args.host}:{args.port}{CHAT_PATH}', file=sys.stderr)
    server.serve_forever()


if __name__ == '__main__':
    main()
