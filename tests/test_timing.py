import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from continugo_train import plot_timing, timing


class TimingTests(unittest.TestCase):
    def test_duration_survives_interruption_and_excludes_resume_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timing.jsonl"
            with patch.object(timing.time, "perf_counter", side_effect=[10, 15, 1000, 1003]):
                with self.assertRaises(KeyboardInterrupt), timing.measure(path, "training"):
                    raise KeyboardInterrupt
                with timing.measure(path, "training"):
                    pass
            rows = plot_timing.read_events(path)
            self.assertEqual([r["seconds"] for r in rows], [5, 3])
            self.assertEqual([r["completed"] for r in rows], [False, True])
            with path.open("a") as stream:
                stream.write('{"phase":')
            self.assertEqual(len(plot_timing.read_events(path)), 2)
            with timing.measure(path, "export"):
                pass
            self.assertEqual(len(plot_timing.read_events(path)), 3)

    def test_measured_breakdown_does_not_double_count_arena(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(
                {
                    "config": {"model_store": str(root / "store")},
                    "iterations": [{"number": 1, "completed": True}],
                },
                root / "loop.pt",
            )
            journal = timing.journal(root, 1)
            journal.parent.mkdir()
            phases = {
                "selfplay": 10,
                "training": 20,
                "window": 1,
                "export": 5,
                "saving": 1,
                "evaluation": 30,
                "arena": 12,
                "iteration": 70,
            }
            journal.write_text(
                "".join(
                    json.dumps({"phase": p, "seconds": seconds, "completed": True}) + "\n"
                    for p, seconds in phases.items()
                )
            )
            rows = plot_timing.collect(root)
            self.assertEqual(rows[0].seconds["arena"], 12)
            self.assertEqual(rows[0].seconds["reporting"], 18)
            self.assertEqual(rows[0].seconds["other"], 3)
            self.assertEqual(sum(rows[0].seconds.values()), 70)
            self.assertFalse(rows[0].estimated)
            self.assertTrue(rows[0].complete)
            plot_timing.plot(rows, root / "timing.png")
            self.assertGreater((root / "timing.png").stat().st_size, 1000)

    def test_historical_estimates_are_explicit_and_incomplete_rounds_omitted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch.save(
                {
                    "config": {"model_store": str(root / "store")},
                    "iterations": [
                        {"number": 1, "completed": True},
                        {"number": 2, "completed": False},
                    ],
                },
                root / "loop.pt",
            )
            log = root / "iteration-0001/train-attempt-0001.log"
            log.parent.mkdir()
            log.write_text("old log")
            with patch.object(plot_timing, "log_spans", return_value={log: 90}):
                rows = plot_timing.collect(root)
            self.assertEqual(len(rows), 1)
            self.assertTrue(rows[0].estimated)
            self.assertEqual(rows[0].seconds["training"], 90)
            with self.assertRaisesRegex(ValueError, "no iteration timing evidence"):
                plot_timing.collect(root, estimates=False)

    def test_arena_history_is_attributed_by_family_and_candidate_round(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "ratings").mkdir()
            (root / "ratings/evaluator.log").write_text(
                "command=['arena', '--candidate', '/models/cg9/r0010']\n"
                "arena game 1/10: moves=50 elapsed=12.300s\n"
                "command=['arena', '--candidate', '/models/cg8/r0010']\n"
                "arena game 1/10: moves=50 elapsed=999.000s\n"
            )
            self.assertEqual(dict(plot_timing.arena_history(root, "cg9")), {10: 12.3})

    def test_invalid_durations_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timing.jsonl"
            path.write_text('{"phase":"training","seconds":-1,"completed":true}\n')
            with self.assertRaisesRegex(ValueError, "invalid duration"):
                plot_timing.read_events(path)
