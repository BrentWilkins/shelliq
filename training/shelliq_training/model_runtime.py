"""Install and run ShellIQ's documentation-conditioned model adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path

from huggingface_hub import snapshot_download

from shelliq_training.documentation_cross_encoder import (
    MODEL_ID,
    REVISION,
    DocumentationCrossEncoderRuntime,
)

CHAT_PATH = '/v1/chat/completions'
MAX_REQUEST_BYTES = 1024 * 1024
INDEX_SHA256 = '17b3b2e9feb215f858e893ba4f761088f9edfbd674ec89633c26d4332a6ab64b'
COMPRESSED_INDEX_SHA256 = '6702b70c7f06c18815aad1a06049890f0cf2ed3fd669e8c7207f8d1255a6773e'
CHECKPOINT_SHA256 = '8cdaf55e4df0670452d1eb2bfc32c68bb1ebbbe1ac20cdda3fb3905d32436ee8'
MODEL_FILES = (
    'added_tokens.json',
    'config.json',
    'merges.txt',
    'pytorch_model.bin',
    'README.md',
    'special_tokens_map.json',
    'tokenizer_config.json',
    'vocab.json',
)


def default_model_home() -> Path:
    override = os.environ.get('SHELLIQ_MODEL_HOME')
    if override:
        return Path(override).expanduser()
    data_home = os.environ.get('XDG_DATA_HOME')
    base = Path(data_home).expanduser() if data_home else Path.home() / '.local' / 'share'
    return base / 'shelliq' / 'model'


def default_index_path() -> Path:
    override = os.environ.get('SHELLIQ_INDEX')
    if override:
        return Path(override).expanduser()
    data_home = os.environ.get('XDG_DATA_HOME')
    base = Path(data_home).expanduser() if data_home else Path.home() / '.local' / 'share'
    return base / 'shelliq' / 'index.sqlite'


def documentation_index_path(home: Path) -> Path:
    return home / 'documentation-index-v2.jsonl'


def checkpoint_path(home: Path) -> Path:
    return home / 'documentation-cross-encoder-v2.pt'


def install_assets(home: Path) -> tuple[Path, Path]:
    home.mkdir(parents=True, exist_ok=True)
    package_assets = resources.files('shelliq_training').joinpath('assets')
    compressed = package_assets.joinpath('documentation-index-v2.jsonl.xz')
    checkpoint = package_assets.joinpath('documentation-cross-encoder-v2.pt')
    if _resource_sha256(compressed) != COMPRESSED_INDEX_SHA256:
        raise RuntimeError('packaged documentation index has an unexpected SHA-256')
    if _resource_sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError('packaged ranker checkpoint has an unexpected SHA-256')

    index_target = documentation_index_path(home)
    if not index_target.exists() or _sha256(index_target) != INDEX_SHA256:
        temporary = index_target.with_name(f'{index_target.name}.tmp-{os.getpid()}')
        with compressed.open('rb') as source, lzma.open(source) as decompressed, temporary.open('wb') as output:
            shutil.copyfileobj(decompressed, output)
        if _sha256(temporary) != INDEX_SHA256:
            temporary.unlink(missing_ok=True)
            raise RuntimeError('decompressed documentation index has an unexpected SHA-256')
        temporary.replace(index_target)

    checkpoint_target = checkpoint_path(home)
    if not checkpoint_target.exists() or _sha256(checkpoint_target) != CHECKPOINT_SHA256:
        temporary = checkpoint_target.with_name(f'{checkpoint_target.name}.tmp-{os.getpid()}')
        with checkpoint.open('rb') as source, temporary.open('wb') as output:
            shutil.copyfileobj(source, output)
        temporary.replace(checkpoint_target)
    return index_target, checkpoint_target


def install_base_model(*, local_files_only: bool = False) -> Path:
    snapshot = snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        allow_patterns=list(MODEL_FILES),
        local_files_only=local_files_only,
    )
    return Path(snapshot)


def load_runtime(home: Path, device: str) -> DocumentationCrossEncoderRuntime:
    index, checkpoint = install_assets(home)
    model = install_base_model(local_files_only=True)
    runtime = DocumentationCrossEncoderRuntime(index, checkpoint, device=device, model_path=model)
    runtime.warm()
    return runtime


def handler(runtime: DocumentationCrossEncoderRuntime, *, quiet: bool = False) -> type[BaseHTTPRequestHandler]:
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
            if not quiet:
                print(f'shelliq-model: {message % args}', file=sys.stderr)

    return Handler


def find_shelliq(requested: Path | None) -> Path:
    if requested is not None:
        candidate = requested.expanduser()
    elif embedded := os.environ.get('SHELLIQ_MODEL_SHELLIQ'):
        candidate = Path(embedded).expanduser()
    elif executable := shutil.which('shelliq'):
        candidate = Path(executable)
    else:
        candidate = Path(__file__).resolve().parents[2] / 'target' / 'release' / 'shelliq'
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise RuntimeError('shelliq executable not found; install a model-enabled release or pass --shelliq')
    return candidate


def scan_index(shelliq: Path, index: Path) -> None:
    subprocess.run([str(shelliq), '--index', str(index), 'index', 'scan'], check=True)


def run_suggestion(args: argparse.Namespace) -> int:
    shelliq = find_shelliq(args.shelliq)
    index = args.index.expanduser()
    if not index.exists():
        print(f'Building local documentation index at {index}', file=sys.stderr)
        scan_index(shelliq, index)
    runtime = load_runtime(args.home.expanduser(), args.device)
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler(runtime, quiet=True))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f'http://127.0.0.1:{server.server_port}{CHAT_PATH}'
    command = [
        str(shelliq),
        '--index',
        str(index),
        'suggest',
        '--endpoint',
        endpoint,
        '--timeout-ms',
        str(args.timeout_ms),
    ]
    if args.json:
        command.append('--json')
    command.extend(args.instruction)
    try:
        return subprocess.run(command, check=False).returncode
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _resource_sha256(resource: resources.abc.Traversable) -> str:
    digest = hashlib.sha256()
    with resource.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(home=default_model_home())
    subparsers = parser.add_subparsers(dest='command', required=True)

    setup = subparsers.add_parser('setup', help='install frozen assets and the pinned CodeT5 base model')
    setup.add_argument('--home', type=Path, default=default_model_home())
    setup.add_argument('--offline', action='store_true', help='require the pinned base model to be cached already')
    setup.add_argument('--scan', action='store_true', help='also scan installed man pages into the ShellIQ index')
    setup.add_argument('--shelliq', type=Path)
    setup.add_argument('--index', type=Path, default=default_index_path())

    serve = subparsers.add_parser('serve', help='run the loopback model adapter')
    serve.add_argument('--home', type=Path, default=default_model_home())
    serve.add_argument('--host', choices=('127.0.0.1', '::1'), default='127.0.0.1')
    serve.add_argument('--port', type=int, default=8080)
    serve.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')

    run = subparsers.add_parser('run', help='start a temporary adapter and run one ShellIQ suggestion')
    run.add_argument('--home', type=Path, default=default_model_home())
    run.add_argument('--shelliq', type=Path)
    run.add_argument('--index', type=Path, default=default_index_path())
    run.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    run.add_argument('--timeout-ms', type=int, default=10_000)
    run.add_argument('--json', action='store_true')
    run.add_argument('instruction', nargs='+')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == 'setup':
        index, checkpoint = install_assets(args.home.expanduser())
        model = install_base_model(local_files_only=args.offline)
        print(f'Documentation index: {index}')
        print(f'Ranker checkpoint: {checkpoint}')
        print(f'Pinned CodeT5 model: {model}')
        if args.scan:
            scan_index(find_shelliq(args.shelliq), args.index.expanduser())
        return 0
    if args.command == 'serve':
        if not 1 <= args.port <= 65535:
            raise SystemExit('--port must be from 1 through 65535')
        runtime = load_runtime(args.home.expanduser(), args.device)
        server = ThreadingHTTPServer((args.host, args.port), handler(runtime))
        print(f'ShellIQ model listening on http://{args.host}:{args.port}{CHAT_PATH}', file=sys.stderr)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            return 130
        finally:
            server.server_close()
    if args.command == 'run':
        return run_suggestion(args)
    raise AssertionError(args.command)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f'shelliq-model: {error}', file=sys.stderr)
        raise SystemExit(1) from None
