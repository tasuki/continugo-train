"""Parallel metadata inspection preserves serial corpus semantics."""

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from continugo_train import data


class CorpusInspectionTests(unittest.TestCase):
    def test_parallel_inspection_is_bounded_and_preserves_order_and_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"{i}.sgf" for i in range(6)]
            for path in paths:
                path.touch()
            barrier = threading.Barrier(3)
            lock = threading.Lock()
            active = peak = 0

            def inspect(path, density, binary):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                try:
                    barrier.wait(timeout=10)
                    return metadata(path)
                finally:
                    with lock:
                        active -= 1

            def metadata(path):
                index = int(path.stem)
                return {
                    "game_id": str(index // 2),  # Recovery copies deduplicate.
                    "file_sha256": str(index),
                    "board_size": 9,
                    "diameter": 9,
                    "resolution": 9,
                    "eligible_plies": [0, 1],
                    "moves": 2,
                    "labeled": True,
                }

            serial_positions, parallel_positions = {}, {}
            with patch.object(data, "inspect_game", side_effect=lambda p, *_: metadata(p)):
                serial = data.inspect_corpus(
                    [str(p) for p in paths],
                    4,
                    workers=1,
                    generated_positions=serial_positions,
                )
            with patch.object(data, "inspect_game", side_effect=inspect):
                parallel = data.inspect_corpus(
                    [str(p) for p in reversed(paths)],
                    4,
                    workers=3,
                    generated_positions=parallel_positions,
                )
            self.assertEqual(peak, 3)
            self.assertEqual(parallel, serial)
            self.assertEqual(parallel_positions, serial_positions)
            self.assertEqual(parallel[1]["duplicate_game"], 3)

    def test_inspection_failure_propagates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.sgf"
            path.touch()
            with (
                patch.object(data, "inspect_game", side_effect=ValueError("bad game")),
                self.assertRaisesRegex(ValueError, "bad game"),
            ):
                data.inspect_corpus([str(path)], 4, workers=2)

    def test_worker_count_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "workers must be positive"):
            data.inspect_corpus([], 4, workers=0)
