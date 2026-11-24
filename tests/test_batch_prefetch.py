"""Bounded full-batch prefetch preserves order and unwinds nested loaders."""

import threading
import unittest
from contextlib import closing
from unittest.mock import patch

from continugo_train import data


class BatchPrefetchTests(unittest.TestCase):
    def assert_stopped(self):
        self.assertFalse(any(t.name == "training-batch-prefetch" for t in threading.enumerate()))

    def test_order_empty_source_and_step_limit(self):
        for count, limit in [(0, 0), (10, 0), (10, 3)]:
            consumed, closed = [], []

            def source(count=count, consumed=consumed, closed=closed):
                try:
                    for index in range(count):
                        consumed.append(index)
                        yield index
                finally:
                    closed.append(True)

            expected = list(range(min(count, limit) if limit else count))
            self.assertEqual(list(data.prefetch_batches(source(), limit=limit)), expected)
            self.assertEqual(consumed, expected)
            self.assertEqual(closed, [True])
            self.assert_stopped()

    def test_queue_is_bounded_and_early_close_releases_backpressure(self):
        consumed, closed = [], []
        third_started = threading.Event()

        def source():
            try:
                for index in range(100):
                    consumed.append(index)
                    if index == 2:
                        third_started.set()
                    yield index
            finally:
                closed.append(True)

        with closing(data.prefetch_batches(source())) as stream:
            self.assertEqual(next(stream), 0)
            self.assertTrue(third_started.wait(timeout=5))
            # One consumed, one queued, and at most one awaiting enqueue.
            self.assertLessEqual(len(consumed), 3)
        self.assertEqual(closed, [True])
        self.assert_stopped()

    def test_producer_errors_reach_consumer_and_close_source(self):
        for error in (ValueError("loader failed"), SystemExit("producer exited")):
            closed = []

            def source(error=error, closed=closed):
                try:
                    yield None
                    raise error
                finally:
                    closed.append(True)

            with closing(data.prefetch_batches(source())) as stream:
                self.assertIsNone(next(stream))
                with self.assertRaises(type(error)) as caught:
                    next(stream)
                self.assertIs(caught.exception, error)
            self.assertEqual(closed, [True])
            self.assert_stopped()

    def test_consumer_exception_closes_nested_parallel_loaders(self):
        closed = []

        def serial(shard, *_):
            try:
                for game, plies in shard:
                    for ply in plies:
                        yield game, ply
            finally:
                closed.append(True)

        plans = [(i, list(range(100))) for i in range(4)]
        with (
            patch.object(data, "_serial_samples", serial),
            self.assertRaisesRegex(RuntimeError, "consumer failed"),
        ):
            source = data._parallel_samples(plans, 4, 2, None)
            with closing(data.prefetch_batches(source)) as stream:
                next(stream)
                raise RuntimeError("consumer failed")
        self.assertEqual(len(closed), 2)
        self.assert_stopped()


if __name__ == "__main__":
    unittest.main()
