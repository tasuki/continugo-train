"""Local browser play: HTTP frontend plus one persistent native move worker."""

import argparse
import functools
import json
import subprocess
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .evaluation import EvaluationConfig
from .runtime import inference_environment
from .selfplay import SelfplayConfig, arguments

ROOT = Path(__file__).resolve().parents[2]


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/api/bot":
            return super().do_GET()
        body = json.dumps({"available": True, "simulations": self.server.simulations}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/api/move":
            self.send_error(404)
            return
        if self.headers.get("Origin") not in (None, f"http://{self.headers.get('Host')}"):
            self.send_error(403)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 2_000_000:
                raise ValueError("Invalid request size")
            request = json.loads(self.rfile.read(length))
            with self.server.worker_lock:
                worker = self.server.worker
                worker.stdin.write(json.dumps(request) + "\n")
                worker.stdin.flush()
                line = worker.stdout.readline()
                if not line:
                    raise ValueError("Move worker stopped; restart the server")
                response = json.loads(line)
            status = 400 if "error" in response else 200
        except (ValueError, OSError) as error:
            status, response = 400, {"error": str(error)}
        body = json.dumps(response).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--simulations", type=int, default=EvaluationConfig.gate_simulations)
    parser.add_argument("--port", type=int, default=9072)
    args = parser.parse_args()
    library, environment = inference_environment(args.device)
    subprocess.run(["cargo", "build", "--locked", "--release", "--bin", "play"], cwd=ROOT, check=True)
    frontend = ROOT.parent / "continugo/frontend"
    subprocess.run(["mise", "run", "build"], cwd=frontend, check=True)
    command = [str(ROOT / "target/release/play"), "--model", str(args.model.resolve()),
               "--device", args.device, "--ort-library", str(library)]
    config = SelfplayConfig(simulations=args.simulations)
    command += arguments(config.evaluation_search() | {
        name: getattr(config, name)
        for name in ("inference_threads", "feature_threads", "cache_positions")
    })
    with subprocess.Popen(command, env=environment, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, text=True, bufsize=1) as worker:
        try:
            handler = functools.partial(Handler, directory=str(frontend / "public"))
            with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
                server.simulations = args.simulations
                server.worker = worker
                server.worker_lock = threading.Lock()
                print(f"Play at http://127.0.0.1:{args.port}/", flush=True)
                server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            worker.terminate()


if __name__ == "__main__":
    main()
