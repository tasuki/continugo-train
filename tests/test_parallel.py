"""Resource-based worker budgeting, independent of any particular workload."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from continugo_train import parallel


class WorkerCountTests(unittest.TestCase):
    def test_process_cpu_count_and_job_limit(self):
        with patch.object(parallel, "os", SimpleNamespace(process_cpu_count=lambda: 12)):
            self.assertEqual(parallel.worker_count(100), 12)
            self.assertEqual(parallel.worker_count(3), 3)
            self.assertEqual(parallel.worker_count(0), 0)

    def test_affinity_not_host_cpu_count(self):
        system = SimpleNamespace(sched_getaffinity=lambda pid: {2, 3}, cpu_count=lambda: 128)
        with patch.object(parallel, "os", system):
            self.assertEqual(parallel.worker_count(10), 2)

    def test_fallback_and_unknown_count(self):
        for cpus, expected in [(8, 8), (None, 1)]:
            with patch.object(parallel, "os", SimpleNamespace(cpu_count=lambda cpus=cpus: cpus)):
                self.assertEqual(parallel.worker_count(20), expected)

    def test_unknown_process_count(self):
        with patch.object(parallel, "os", SimpleNamespace(process_cpu_count=lambda: None)):
            self.assertEqual(parallel.worker_count(10), 1)

    def test_negative_job_count(self):
        with self.assertRaises(ValueError):
            parallel.worker_count(-1)
