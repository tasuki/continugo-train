"""cg13 generation, availability ramp-up and physical/logical batch contracts."""

import copy
import unittest
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import torch
from test_loop import FakeJobs, LoopHarness

from continugo_train.data import Batch, _plans
from continugo_train.loop import LoopConfig, LoopRunner, parse_args
from continugo_train.sampling import available_position_quotas
from continugo_train.train import Options, microbatches, run_phase

SIZES = (9, 13, 17, 11, 15, 19)
WEIGHTS = "9:1,11:1,13:1,15:1,17:1,19:1"
COUNTS = "9:150,11:123,13:104,15:90,17:79,19:71"


def config():
    return LoopConfig(
        sizes=SIZES, size_weights=WEIGHTS, size_schedule="iteration",
        games_by_size=COUNTS, size_sampling="available", microbatch_size_9=64,
    )


def game(size, count):
    return SimpleNamespace(board_size=size, plies=tuple(range(count)), game_id=f"{size:016x}")


def batch(count=5, width=38):
    return Batch(
        coarse=torch.linspace(0, 1, count)[:, None, None, None].expand(count, 1, width, width),
        exact=torch.zeros(count, 1, 1), legal=torch.ones(count, 1, dtype=torch.bool),
        actions=torch.zeros(count, 1, dtype=torch.int64),
        indices=torch.zeros(count, 1, dtype=torch.int64), visits=torch.ones(count, 1),
        outcomes=torch.ones(count), board_sizes=torch.full((count,), 9),
        scores=torch.linspace(-1, 1, count), ownership=torch.zeros(count, width, width),
        policy_weights=torch.tensor([0., 0., 1., 1., 0.])[:count], komi=torch.zeros(count),
    )


class TinyNet(torch.nn.Module):
    """No BatchNorm: tests loss normalization independently of BN microbatch effects."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.2))

    def forward_all(self, coarse, exact, actions, legal, komi):
        x = coarse[:, 0, 0, 0] * self.weight
        policy = torch.stack((x, -x), dim=1).log_softmax(dim=1)
        return policy, x.tanh(), x, x[:, None, None].expand(-1, *coarse.shape[-2:]), x.exp()


class MixedBoardTests(unittest.TestCase):
    def test_iteration_counts_cycle_seeds_and_cli(self):
        c = config()
        c.validate()
        expected = (150, 104, 79, 123, 90, 71)
        runner = object.__new__(LoopRunner)
        runner.config = c
        seeds = []
        for number in range(1, 19):
            self.assertEqual(c.game_count(number), expected[(number - 1) % 6])
            for index in range(c.game_count(number)):
                size, seed = runner.plan({"number": number}, index)
                self.assertEqual(size, SIZES[(number - 1) % 6])
                seeds.append(seed)
        self.assertEqual(seeds, list(range(sum(expected) * 3)))
        # Partial-job adoption uses the original index, never reseeds remaining games.
        self.assertEqual(runner.plan({"number": 2}, 17), (13, 167))
        args = parse_args([
            "--sizes", ",".join(map(str, SIZES)), "--size-schedule", "iteration",
            "--games-by-size", COUNTS, "--size-sampling", "available",
        ])
        self.assertEqual(args.sizes, SIZES)
        self.assertEqual(args.games_by_size, COUNTS)
        for invalid in (
            replace(c, games_by_size="9:150"), replace(c, games_by_size="uniform"),
            replace(c, size_schedule="mixed"), replace(c, size_weights="9:1,13:0"),
            replace(c, microbatch_size_9=-1), replace(c, size_sampling="bad"),
        ):
            with self.assertRaises(ValueError):
                invalid.validate()

    def test_old_mixed_generation_is_unchanged(self):
        c = LoopConfig(sizes=(9, 13, 19), size_weights="9:2,13:2,19:1", games_per_iteration=3)
        runner = object.__new__(LoopRunner)
        runner.config = c
        planned = [runner.plan({"number": n}, i) for n in (1, 2) for i in range(3)]
        self.assertEqual(planned, list(zip((9, 13, 19, 9, 13, 9), range(6))))

    def test_capped_equal_shares_redistribute_and_reach_equal_target(self):
        self.assertEqual(available_position_quotas({9: 1000}, WEIGHTS, 60), {9: 60})
        self.assertEqual(available_position_quotas({9: 1000, 13: 5}, WEIGHTS, 60), {9: 55, 13: 5})
        all_sizes = {size: 1000 for size in SIZES}
        self.assertEqual(available_position_quotas(all_sizes, WEIGHTS, 60), dict.fromkeys(SIZES, 10))
        self.assertEqual(available_position_quotas({9: 3, 13: 2}, WEIGHTS, 60), {9: 3, 13: 2})
        self.assertEqual(available_position_quotas({9: 3, 13: 2}, "9:1,13:0", 60), {9: 3})
        quotas = available_position_quotas(dict.fromkeys(SIZES, 100_000), WEIGHTS, 65_536)
        self.assertEqual(sum(quotas.values()), 65_536)
        self.assertLessEqual(max(quotas.values()) - min(quotas.values()), 1)

    def test_plans_are_unique_seeded_refreshing_and_validation_exhaustive(self):
        games = [game(9, 1000), game(13, 5), game(19, 1000)]

        def plans(epoch=1, training=True):
            return _plans(games, 42, epoch, training, 0, WEIGHTS, 8, "available", 65)

        selected = plans()
        references = [(g.board_size, p) for g, plies in selected for p in plies]
        self.assertEqual(Counter(size for size, _ in references), {9: 30, 13: 5, 19: 30})
        self.assertEqual(len(set(references)), 65)
        self.assertEqual(selected, plans())
        self.assertNotEqual(selected, plans(2))
        self.assertEqual(sum(len(plies) for _, plies in plans(training=False)), 2005)
        self.assertEqual(plans(training=False), plans(2, training=False))

    def test_microbatch_area_limit_metadata_and_close(self):
        source = batch()
        pieces = list(microbatches(iter([source]), 2))
        self.assertEqual([len(b.outcomes) for b, *_ in pieces], [2, 2, 1])
        self.assertEqual([p[3:] for p in pieces], [(True, False), (False, False), (False, True)])
        self.assertTrue(all(p[1:3] == (5, 2) for p in pieces))
        torch.testing.assert_close(torch.cat([p[0].coarse for p in pieces]), source.coarse)
        self.assertEqual(len(list(microbatches(iter([batch(width=78)]), 2))), 5)
        self.assertEqual(len(list(microbatches(iter([source]), 0))), 1)
        closed = []

        def producer():
            try:
                yield source
                self.fail("must not fetch another logical batch")
            finally:
                closed.append(True)

        stream = microbatches(producer(), 2)
        next(stream)
        stream.close()
        self.assertEqual(closed, [True])

    def test_gradient_accumulation_preserves_policy_and_other_head_denominators(self):
        torch.set_num_threads(1)
        initial = TinyNet()
        results = []
        for limit in (0, 2):
            model = copy.deepcopy(initial)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            with patch("continugo_train.train.batches", side_effect=lambda *a, **k: iter([batch()])):
                metrics = run_phase(
                    model, optimizer, [game(9, 5)],
                    Options(batch_size=5, microbatch_size_9=limit, grad_clip=0, steps_per_epoch=1),
                    1, torch.device("cpu"), None, True,
                )
            self.assertEqual(metrics["steps"], 1)
            self.assertEqual(metrics["samples"], 5)
            self.assertEqual(metrics["policy_samples"], 2)
            self.assertEqual(metrics["replay_exposures_by_size"], {9: 1.0})
            results.append((model, optimizer, metrics))
        torch.testing.assert_close(results[0][0].weight, results[1][0].weight)
        torch.testing.assert_close(
            results[0][1].state_dict()["state"][0]["momentum_buffer"],
            results[1][1].state_dict()["state"][0]["momentum_buffer"],
        )
        self.assertAlmostEqual(results[0][2]["loss"], results[1][2]["loss"], places=6)
        self.assertEqual(results[1][2]["physical_batches"], 3)


class MixedJobs(FakeJobs):
    def selfplay(self, command):
        size = int(command[command.index("--sizes") + 1])
        converted = list(command)
        converted[converted.index("--sizes") + 1] = "13"
        output = command[command.index("--output") + 1]
        try:
            super().selfplay(converted)
        finally:
            from pathlib import Path

            for path in Path(output).glob("game-*-size13-*.sgf"):
                path.write_text(path.read_text().replace("SZ[13]", f"SZ[{size}]").replace(
                    "RE[B+169.0]", f"RE[B+{size * size}.0]"
                ))
                path.rename(path.with_name(path.name.replace("size13-", f"size{size}-")))


class MixedLoopRecoveryTests(LoopHarness):
    def test_old_loop_and_uncommitted_epoch_restore_legacy_defaults(self):
        output = self.root / "old"
        self.jobs.fault = "epoch_gap"
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop(output)
        state = torch.load(output / "loop.pt", weights_only=True)
        for name in ("size_schedule", "games_by_size", "size_sampling", "microbatch_size_9"):
            del state["config"][name]
        torch.save(state, output / "loop.pt")
        epoch = next(output.glob("iteration-*/train-attempt-*/epoch-0001.pt"))
        saved = torch.load(epoch, weights_only=True)
        for name in ("size_sampling", "microbatch_size_9"):
            del saved["options"][name]
        torch.save(saved, epoch)
        calls = len(self.stage_calls("continugo_train.train"))
        resumed = self.run_loop(output, resume=True)
        self.assertTrue(resumed["iterations"][0]["completed"])
        self.assertEqual(resumed["config"]["size_schedule"], "mixed")
        self.assertEqual(resumed["config"]["size_sampling"], "downsample")
        self.assertEqual(len(self.stage_calls("continugo_train.train")), calls)

    def test_six_sizes_variable_counts_and_partial_generation_recovery(self):
        self.jobs = MixedJobs()
        self.jobs.fault = "partial_selfplay"
        output = self.root / "mixed"
        extra = [
            "--iterations", "6", "--sizes", ",".join(map(str, SIZES)),
            "--size-weights", WEIGHTS, "--size-schedule", "iteration",
            "--games-by-size", "9:3,11:2,13:2,15:2,17:2,19:2",
            "--size-sampling", "available", "--microbatch-size-9", "2",
        ]
        with self.assertRaises(KeyboardInterrupt):
            self.run_loop(output, extra=extra)
        state = self.run_loop(output, resume=True)
        seeds = []
        for iteration, size, count in zip(state["iterations"], SIZES, (3, 2, 2, 2, 2, 2)):
            self.assertTrue(iteration["completed"])
            self.assertEqual(len(iteration["games"]), count)
            self.assertEqual({r["board_size"] for r in iteration["games"]}, {size})
            seeds.extend(r["seed"] for r in iteration["games"])
            saved = torch.load(output / iteration["checkpoint"]["path"], weights_only=True)
            self.assertEqual(saved["options"]["size_sampling"], "available")
            self.assertEqual(saved["options"]["microbatch_size_9"], 2)
        self.assertEqual(seeds, list(range(204, 217)))
        self.assertEqual(len(state["iterations"][0]["selfplay_attempts"]), 2)
        calls = len(self.jobs.calls)
        self.run_loop(output, resume=True)
        self.assertEqual(len(self.jobs.calls), calls)
        with self.assertRaisesRegex(ValueError, "conflicting override"):
            self.run_loop(output, resume=True, extra=["--size-schedule", "mixed"])


if __name__ == "__main__":
    unittest.main()
