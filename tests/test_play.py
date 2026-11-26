"""The browser bridge uses one JSON line per request, including worker errors."""
import functools
import io
import json
import threading
import tempfile
import unittest
from pathlib import Path
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

from continugo_train.play import Handler


class PlayTests(unittest.TestCase):
    def test_browser_bridge(self):
        with tempfile.TemporaryDirectory() as directory:
            check_browser_bridge(Path(directory))


def check_browser_bridge(tmp_path):
    (tmp_path / "index.html").write_text("board")
    worker = SimpleNamespace(
        stdin=io.StringIO(),
        stdout=io.StringIO('{"move":null}\n{"error":"Game is finished"}\n'),
    )
    with ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(tmp_path))) as server:
        server.worker = worker
        server.worker_lock = threading.Lock()
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        client = HTTPConnection(*server.server_address)
        try:
            client.request("GET", "/")
            response = client.getresponse()
            assert response.status == 200
            assert response.read() == b"board"
            client.request("GET", "/api/bot")
            response = client.getresponse()
            assert response.status == 200
            assert json.loads(response.read()) == {"available": True, "simulations": 128}
            request = {"sgf": "(;FF[9071]GM[9071]SZ[9]KM[7])", "ply": 0}
            for status, expected in [(200, {"move": None}), (400, {"error": "Game is finished"})]:
                client.request("POST", "/api/move", json.dumps(request))
                response = client.getresponse()
                assert response.status == status
                assert json.loads(response.read()) == expected
            assert [json.loads(line) for line in worker.stdin.getvalue().splitlines()] == [request, request]
            client.request("POST", "/api/move", "not json")
            response = client.getresponse()
            assert response.status == 400
            assert "error" in json.loads(response.read())
        finally:
            client.close()
            server.shutdown()
            thread.join()
