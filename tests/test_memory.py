import subprocess
import unittest
from unittest.mock import patch

from continugo_train.memory import GpuMemorySampler, descendant


class MemoryTests(unittest.TestCase):
    def test_parent_walk_and_missing_process(self):
        with patch(
            "continugo_train.memory.Path.read_text", side_effect=["PPid:\t20\n", "PPid:\t10\n"]
        ):
            self.assertTrue(descendant(30, 10))
        with patch("continugo_train.memory.Path.read_text", side_effect=FileNotFoundError):
            self.assertFalse(descendant(30, 10))
        with patch("continugo_train.memory.Path.read_text", return_value="PPid:\t30\n"):
            self.assertFalse(descendant(30, 10))

    def test_gpu_memory_sums_only_the_tree_across_devices(self):
        sampler = GpuMemorySampler(10, enabled=False)
        result = subprocess.CompletedProcess([], 0, "10, 100\n20, 30\n99, 9000\n10, 50\n")
        with (
            patch("continugo_train.memory.subprocess.run", return_value=result),
            patch(
                "continugo_train.memory.descendant", side_effect=lambda pid, root: pid in (10, 20)
            ),
        ):
            sampler._sample()
            self.assertEqual(sampler.summary()["sampled_gpu_peak_mib"], 180)
            self.assertEqual(sampler.summary()["gpu_memory_samples"], 1)
            result.stdout = "10, 1\n"
            sampler._sample()
            self.assertEqual(sampler.summary()["sampled_gpu_peak_mib"], 180)
            result.stdout = "10, [N/A]\n"
            sampler._sample()
            self.assertEqual(sampler.summary()["gpu_memory_samples"], 2)

    def test_unavailable_monitor_is_null_and_does_not_launch(self):
        with patch("continugo_train.memory.subprocess.run") as run:
            with GpuMemorySampler(enabled=False) as sampler:
                pass
            self.assertIsNone(sampler.summary()["sampled_gpu_peak_mib"])
            self.assertEqual(sampler.summary()["gpu_memory_samples"], 0)
            run.assert_not_called()
