import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from continugo_train.plot_metrics import (
    collect_draws,
    collect_jobs,
    main,
    plot,
    plot_draws,
    read_epochs,
    update,
)


def epoch(number, loss=3, *, saved=True):
    text = (
        f"epoch={number}/4 train: loss={loss} policy_ce=2 value_mse=1 score_mse=0.2 ownership_mse=0.3 samples=2\n"
        f"epoch={number}/4 validation: loss={loss + 0.5} policy_ce=2.5 value_mse=1 score_mse=0.2 ownership_mse=0.3 samples=2\n"
    )
    if saved:
        text += f"saved /some/job/epoch-{number:04d}.pt; network=test\n"
    return text


class PlotMetricsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def log(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def test_saved_epochs_only_and_resumed_epoch_replacement(self):
        path = self.log(
            "training.log",
            "options=anything\n"
            + epoch(1)
            + epoch(2, 8, saved=False)
            + epoch(2, 4)
            + epoch(3, saved=False)
            + "epoch=4/4 train: loss=",
        )
        values = read_epochs(path)
        self.assertEqual(list(values), [1, 2])
        self.assertEqual(values[2]["train"]["loss"], 4)
        self.assertEqual(values[2]["validation"]["value_mse"], 1)
        self.assertEqual(collect_jobs(self.root)[0].label, "Training")

    def test_campaign_order_recovery_attempts_and_active_round(self):
        self.log("initial-training/training.log", epoch(1) + epoch(2))
        self.log("loop/iteration-0010/train-attempt-0001/training.log", epoch(1))
        self.log("loop/iteration-0002/train-attempt-0001/training.log", epoch(1, 8))
        self.log("loop/iteration-0002/train-attempt-0002/training.log", epoch(1, 4) + epoch(2))
        self.log("loop/iteration-0011/train-attempt-0001/training.log", epoch(1, saved=False))
        # Unrelated snapshots/logs are not recursive plotting inputs.
        self.log("archive/training.log", epoch(1))
        jobs = collect_jobs(self.root)
        self.assertEqual([job.label for job in jobs], ["Bootstrap", "Round 2", "Round 10"])
        self.assertEqual([len(job.epochs) for job in jobs], [2, 2, 1])
        self.assertEqual(jobs[1].epochs[1]["train"]["loss"], 4)
        self.assertEqual(len(collect_jobs(self.root / "loop")), 2)

    def test_bad_metrics_missing_phases_and_empty_input_are_errors(self):
        for text in [
            epoch(1).replace("loss=3 ", "loss=nan "),
            epoch(1).replace("policy_ce=2 ", "policy_ce=oops "),
            "saved /job/epoch-0001.pt; network=test\n",
        ]:
            with self.subTest(text=text):
                path = self.log("training.log", text)
                with self.assertRaises(ValueError):
                    read_epochs(path)
        path.write_text(epoch(1, saved=False))
        with self.assertRaisesRegex(ValueError, "no saved-epoch"):
            collect_jobs(self.root)
        with self.assertRaisesRegex(ValueError, "does not exist"):
            collect_jobs(self.root / "missing")

    def draw_state(self):
        loop = self.root / "loop"
        loop.mkdir()
        state = {
            "iterations": [
                # Repeated actual games must count separately (unlike replay).
                {"number": 1, "games": [{"result": "0"}, {"result": "0"}, {"result": "B+1"}]},
                {"number": 2, "games": [{"result": "W+2"}, {"result": None}]},
                {"number": 3, "games": [{"result": "0"}, None]},
                {"number": 4, "games": []},
            ],
            "initial_reference": {"games": [{"result": "0"}]},
        }
        torch.save(state, loop / "loop.pt")
        return loop

    def test_draws_use_complete_selfplay_batches_and_keep_truncations_unknown(self):
        self.assertEqual(collect_draws(self.root), [])
        loop = self.draw_state()
        rates = collect_draws(self.root)
        self.assertEqual(
            [(r.iteration, r.games, r.draws, r.truncated) for r in rates],
            [
                (1, 3, 2, 0),
                (2, 2, 0, 1),
            ],
        )
        self.assertEqual(collect_draws(loop), rates)

    def test_auto_update_targets_campaign_from_epoch_directory(self):
        loop = self.draw_state()
        epoch_dir = loop / "iteration-0001/train-attempt-0001"
        self.log("loop/iteration-0001/train-attempt-0001/training.log", epoch(1))
        with (
            patch("continugo_train.plot_metrics.plot") as losses,
            patch("continugo_train.plot_metrics.plot_draws") as draws,
        ):
            update(epoch_dir)
        self.assertEqual(losses.call_args.args[1], self.root / "metrics.png")
        self.assertEqual(draws.call_args.args[1], self.root / "draws.png")

    def test_auto_draw_update_before_first_saved_epoch_and_plot_failure(self):
        loop = self.draw_state()
        with patch("continugo_train.plot_metrics.plot_draws") as draws:
            update(loop)
        draws.assert_called_once()
        with patch("continugo_train.plot_metrics.plot_draws", side_effect=RuntimeError("failed")):
            update(loop)  # Diagnostics cannot stop self-play/training.

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "install plotting extra")
    def test_draw_plot_and_cli_generate_both_without_changing_loop_state(self):
        self.log("initial-training/training.log", epoch(1))
        loop = self.draw_state()
        before = (loop / "loop.pt").read_bytes()
        output = self.root / "draws.svg"
        plot_draws(collect_draws(self.root), output, "Draw rates")
        svg = output.read_text()
        self.assertIn("2/3", svg)
        self.assertIn("Truncated / unknown", svg)
        self.assertEqual(main(["--run", str(self.root)]), 0)
        self.assertTrue((self.root / "draws.png").read_bytes().startswith(b"\x89PNG"))
        self.assertEqual((loop / "loop.pt").read_bytes(), before)
        with self.assertRaisesRegex(ValueError, "no fully recorded"):
            plot_draws([], self.root / "empty.png", "Empty")
        conflict = self.root / "conflict.png"
        self.assertEqual(
            main(
                [
                    "--run",
                    str(self.root),
                    "--output",
                    str(conflict),
                    "--draws-output",
                    str(conflict),
                ]
            ),
            1,
        )
        self.assertFalse(conflict.exists())

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "install plotting extra")
    def test_connects_rounds_but_preserves_missing_epoch_gaps(self):
        import math
        import matplotlib.pyplot as plt

        self.log("loop/iteration-0001/train-attempt-0001/training.log", epoch(1, 3))
        self.log("loop/iteration-0002/train-attempt-0001/training.log", epoch(1, 7) + epoch(3, 5))
        with patch("continugo_train.plot_metrics.save_figure") as save:
            plot(collect_jobs(self.root), self.root / "metrics.png", "Connected")
        figure = save.call_args.args[1]
        try:
            for axis in figure.axes[:5]:
                lines = {line.get_label(): line for line in axis.lines}
                self.assertIn("Train", lines)
                self.assertIn("Validation", lines)
                for label in ("Train", "Validation"):
                    line = lines[label]
                    self.assertEqual(list(line.get_xdata()), [1, 2, 3, 4])
                    self.assertTrue(math.isnan(line.get_ydata()[2]))
                    self.assertEqual(line.get_linestyle(), "-")
            train = next(line for line in figure.axes[0].lines if line.get_label() == "Train")
            self.assertEqual(list(train.get_ydata()[:2]), [3, 7])
        finally:
            plt.close(figure)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "install plotting extra")
    def test_plot_svg_png_and_cli_read_only(self):
        path = self.log("training.log", epoch(1) + epoch(2, 2))
        original = path.read_bytes()
        jobs = collect_jobs(self.root)
        output = self.root / "plots/losses.svg"
        plot(jobs, output, "Test <losses>")
        self.assertIn("<svg", output.read_text())
        self.assertIn("Value MSE", output.read_text())
        self.assertEqual(main(["--run", str(self.root)]), 0)
        self.assertTrue((self.root / "metrics.png").read_bytes().startswith(b"\x89PNG"))
        self.assertEqual(path.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, "must end in"):
            plot(jobs, self.root / "bad.html", "Test")


if __name__ == "__main__":
    unittest.main()
