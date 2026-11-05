import io
from contextlib import redirect_stdout
from unittest.mock import patch

import test_loop as helpers
from test_acceptance import ArenaJobs

from continugo_train import loop, ratings, round_robin


class RoundRobinTests(helpers.LoopHarness):
    def setUp(self):
        super().setUp()
        self.jobs = ArenaJobs()
        self.source_loop = self.root / "training"
        self.source_state = self.run_loop(
            self.source_loop, extra=["--iterations", "2", "--reference-games", "2"]
        )
        self.output = self.root / "round-robin"

    def evaluate(self, resume=False, ingest_only=False):
        args = ["--output", str(self.output)]
        args += (
            ["--resume"]
            if resume
            else ["--loop", str(self.source_loop), "--latest", "3", "--games", "2"]
        )
        if ingest_only:
            args += ["--ingest-only"]
        with patch.object(loop, "run_process", self.jobs), redirect_stdout(io.StringIO()):
            return round_robin.run(round_robin.parse_args(args))

    def test_round_robin_rates_all_pairs_without_changing_training_and_is_idempotent(self):
        before = (self.source_loop / "loop.pt").read_bytes()
        state = self.evaluate()
        self.assertEqual(len(state["jobs"]), 3)
        self.assertEqual(
            len(
                {
                    frozenset(
                        (j["model"]["checkpoint_sha256"], j["parent_model"]["checkpoint_sha256"])
                    )
                    for j in state["jobs"]
                }
            ),
            3,
        )
        with ratings.transaction(self.source_state["config"]["model_store"]) as (tables, _):
            head_to_head = [r for r in tables.matches if r["kind"] == "gate"]
        self.assertEqual(len(head_to_head), 6)
        self.assertTrue(
            all(ratings.REFERENCE_ID not in (r["black"], r["white"]) for r in head_to_head)
        )
        self.assertEqual((self.source_loop / "loop.pt").read_bytes(), before)
        calls = len(self.jobs.calls)
        self.evaluate(resume=True)
        self.assertEqual(len(self.jobs.calls), calls)

    def test_interrupt_adopts_completed_games_and_does_not_replay_them(self):
        self.jobs.arena_mode = "partial"
        with self.assertRaises(KeyboardInterrupt):
            self.evaluate()
        path = next(self.output.glob("iteration-0001/gate/*/game-000001-*.sgf"))
        before = path.read_bytes()
        state = self.evaluate(resume=True)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(state["jobs"][0]["gate"]["attempts"][1]["indices"], [1])

    def test_ingest_only_preserves_partial_evidence_without_launching_games(self):
        self.jobs.arena_mode = "partial"
        with self.assertRaises(KeyboardInterrupt):
            self.evaluate()
        calls = len(self.jobs.calls)
        self.evaluate(resume=True, ingest_only=True)
        self.assertEqual(len(self.jobs.calls), calls)
        with ratings.transaction(self.source_state["config"]["model_store"]) as (tables, _):
            self.assertEqual(sum(r["kind"] == "gate" for r in tables.matches), 1)
        self.evaluate(resume=True, ingest_only=True)
        self.assertEqual(len(self.jobs.calls), calls)
        self.evaluate(resume=True)
        with ratings.transaction(self.source_state["config"]["model_store"]) as (tables, _):
            self.assertEqual(sum(r["kind"] == "gate" for r in tables.matches), 6)

    def test_resume_refuses_a_live_lock_and_changed_graph(self):
        self.evaluate()
        with loop.loop_lock(self.output), self.assertRaisesRegex(ValueError, "locked"):
            self.evaluate(resume=True)
        graph = self.output / "models/0000/encoder.onnx"
        graph.write_text("changed")
        with self.assertRaisesRegex(ValueError, "missing or changed"):
            self.evaluate(resume=True)

    def test_seed_overflow_is_rejected_before_copying_models(self):
        with self.assertRaisesRegex(ValueError, "overflow"):
            round_robin.initialize(self.output, self.source_state, [self.root, self.root], 2, 2**64)

    def test_odd_game_counts_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "even nonnegative"):
            round_robin.initialize(self.output, self.source_state, [self.root, self.root], 3, 1)
