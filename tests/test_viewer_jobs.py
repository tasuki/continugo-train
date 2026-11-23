"""Parallel report scheduling and crash-safe publication."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from continugo_train import viewer_jobs


class ViewerJobsTests(unittest.TestCase):
    def test_serial_atomic_publication_and_thread_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "game.nn.html"
            threads = torch.get_num_threads()

            def render(_sgf, draft, *_args, **_kwargs):
                self.assertEqual(torch.get_num_threads(), 1)
                self.assertFalse(output.exists())
                Path(draft).write_text("complete")

            with (
                patch.object(viewer_jobs, "worker_count", return_value=1),
                patch.object(viewer_jobs.nn_view, "generate_rated", render),
            ):
                viewer_jobs.generate([("game.sgf", output, {}, {}, None)])
            self.assertEqual(output.read_text(), "complete")
            self.assertEqual(torch.get_num_threads(), threads)
            self.assertEqual(list(Path(directory).iterdir()), [output])

    def test_failure_preserves_previous_report_and_cleans_draft(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "game.nn.html"
            output.write_text("previous")
            threads = torch.get_num_threads()

            def fail(_sgf, draft, *_args, **_kwargs):
                Path(draft).write_text("partial")
                raise OSError("disk full")

            with (
                patch.object(viewer_jobs, "worker_count", return_value=1),
                patch.object(viewer_jobs.nn_view, "generate_rated", fail),
                self.assertRaisesRegex(OSError, "disk full"),
            ):
                viewer_jobs.generate([("game.sgf", output, {}, {}, None)])
            self.assertEqual(output.read_text(), "previous")
            self.assertEqual(torch.get_num_threads(), threads)
            self.assertEqual(list(Path(directory).iterdir()), [output])

    def test_pool_is_bounded_spawned_and_propagates_errors(self):
        jobs = [("game.sgf", "game.nn.html", {}, {}, None)] * 10
        with (
            patch.object(viewer_jobs, "worker_count", return_value=2) as budget,
            patch.object(viewer_jobs, "ProcessPoolExecutor") as executor,
        ):
            pool = executor.return_value.__enter__.return_value
            pool.map.return_value = iter([None] * 10)
            viewer_jobs.generate(jobs)
            budget.assert_called_once_with(10)
            options = executor.call_args.kwargs
            self.assertEqual(options["max_workers"], 2)
            self.assertEqual(options["mp_context"].get_start_method(), "spawn")
            self.assertIs(options["initializer"], viewer_jobs._initialize_worker)
            pool.map.assert_called_once_with(viewer_jobs._render, jobs, chunksize=1)
            pool.map.side_effect = RuntimeError("worker failed")
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                viewer_jobs.generate(jobs)

    def test_empty_batch_does_not_start_workers(self):
        with patch.object(viewer_jobs, "ProcessPoolExecutor") as executor:
            viewer_jobs.generate([])
            executor.assert_not_called()
