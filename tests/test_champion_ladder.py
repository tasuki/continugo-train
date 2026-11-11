import io
import json
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import test_loop as helpers
import torch
from test_acceptance import ArenaJobs

from continugo_train import loop, rate_saved, rating_store, saved_models
from continugo_train.evaluation import EvaluationConfig


class ScriptedArena(ArenaJobs):
    def __init__(self):
        super().__init__()
        self.wins = {"reference": 7, "champion-screen": 7, "champion-extension": 21}
        self.interrupt_kind = None

    def __call__(self, command, logfile, lock_fd, environment=None):
        if "--candidate" not in command:
            return super().__call__(command, logfile, lock_fd, environment)
        output = Path(command[command.index("--output") + 1])
        reference = "--reference-random" in command
        end = int(command[command.index("--games") + 1]) + int(
            command[command.index("--game-offset") + 1]
        )
        kind = (
            "champion-extension" if end == 30 else "reference" if reference else "champion-screen"
        )
        if self.interrupt_kind == kind:
            self.interrupt_kind = None
            if reference:
                self.reference_mode = "partial"
            else:
                self.arena_mode = "partial"
        try:
            super().__call__(command, logfile, lock_fd, environment)
        finally:
            for path in output.glob("game-*.sgf"):
                text = path.read_text()
                if "RE[B+220.4]" not in text:
                    continue
                ordinal = int(path.name.split("-")[1]) - 1
                candidate_black = "candidate-black" in path.name
                black_wins = candidate_black == (ordinal < self.wins[kind])
                header = text.split(";B[", 1)[0].replace(
                    "RE[B+220.4]", "RE[B+220.4]" if black_wins else "RE[W+220.4]"
                )
                moves = (
                    [("B", "dSdS"), ("W", ""), ("B", "")]
                    if black_wins
                    else [("B", ""), ("W", "dSdS"), ("B", ""), ("W", "")]
                )
                count = int(command[command.index("--simulations") + 1])
                nodes = []
                for color, coordinate in moves:
                    comment = ""
                    if not reference or (color == "B") == candidate_black:
                        policy = json.dumps({"policy": [[coordinate, count]]}).replace("]", "\\]")
                        comment = f"C[{policy}]"
                    nodes.append(f";{color}[{coordinate}]{comment}")
                path.write_text(header + "".join(nodes) + ")")


class ChampionTests(helpers.LoopHarness):
    def setUp(self):
        super().setUp()
        self.jobs = ScriptedArena()
        self.campaign = self.root / "campaign"
        self.state = self.run_loop(
            self.campaign, extra=["--iterations", "3", "--save-model-every", "1"]
        )
        self.ids = [f"test1-r{i:04d}" for i in range(1, 4)]
        self.store = Path(self.state["config"]["model_store"])
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(
            patch.dict(rate_saved.POLICY, {"evaluation_every": 1, "challenge_every": 1})
        )
        self.stack.enter_context(patch.object(loop, "run_process", self.jobs))

        def report(_sgf, output, *_args, **_kwargs):
            Path(output).write_text("neural report")

        self.stack.enter_context(patch.object(rating_store.nn_view, "generate_rated", report))
        self.stack.enter_context(patch.object(rate_saved.plot_ratings, "update"))
        self.stack.enter_context(redirect_stdout(io.StringIO()))

    def evaluate(self, resume=False):
        args = ["--model-store", str(self.store)]
        args += ["--resume"] if resume else ["--family", "test1"]
        return rate_saved.run(rate_saved.parse_args(args))

    def saved_state(self):
        return torch.load(self.store / "ratings/evaluator.pt", weights_only=True)

    def test_only_every_fifth_round_gets_any_evaluation(self):
        rate_saved.POLICY.update(evaluation_every=5, challenge_every=5)
        self.run_loop(self.campaign, resume=True, extra=["--iterations", "5"])
        self.evaluate()
        registry = rate_saved.load_champions(self.store)
        self.assertEqual(registry["champion"], "test1-r0005")
        self.assertEqual(
            [entry["model"] for entry in registry["history"]], ["random", "test1-r0005"]
        )
        for index in range(1, 5):
            name = f"test1-r{index:04d}"
            self.assertIn(name, registry["skipped"])
            self.assertNotIn(name, registry["evaluated"])
        matches = list(self.saved_state()["matches"].values())
        self.assertEqual(len([job for job in matches if job["kind"] == "reference"]), 1)
        for job in matches:
            if job["kind"].startswith("champion-"):
                self.assertEqual(job["candidate_name"], "test1-r0005")
        self.assertEqual(
            len(
                rate_saved.rating_store.refresh(
                    self.store,
                    EvaluationConfig.from_config(self.saved_state()["config"]),
                    visualize=False,
                )[1]
            ),
            2,
        )

    def test_cadence_uses_round_numbers_not_queue_positions(self):
        rate_saved.POLICY.update(evaluation_every=5, challenge_every=5)
        self.run_loop(self.campaign, resume=True, extra=["--iterations", "5"])
        (self.store / "ratings").mkdir(exist_ok=True)
        state = rate_saved.initialize(
            self.store,
            rate_saved.parse_args(
                [
                    "--model-store",
                    str(self.store),
                    "--family",
                    "test1",
                ]
            ),
        )
        state["candidates"] = ["test1-r0004", "test1-r0005"]
        loop.atomic_checkpoint(self.store / "ratings/evaluator.pt", state)
        self.evaluate(resume=True)
        registry = rate_saved.load_champions(self.store)
        self.assertIn("test1-r0004", registry["skipped"])
        self.assertFalse(
            any(
                job["candidate_name"] == "test1-r0004"
                for job in self.saved_state()["matches"].values()
            )
        )
        self.assertEqual(registry["evaluated"]["test1-r0005"]["decision"], "promoted")
        self.assertEqual(registry["champion"], "test1-r0005")

    def test_partial_off_cycle_batch_is_abandoned_without_new_games(self):
        self.jobs.interrupt_kind = "reference"
        with self.assertRaises(KeyboardInterrupt):
            self.evaluate()
        paths = list((self.store / "sgfs").glob("*/*.sgf"))
        before = {path: path.read_bytes() for path in paths}
        calls = len(self.jobs.calls)
        rate_saved.POLICY.update(evaluation_every=5, challenge_every=5)
        state = self.saved_state()
        registry = json.loads((self.store / "ratings/champions.json").read_text())
        state["policy"] = dict(rate_saved.POLICY)
        registry["policy"] = dict(rate_saved.POLICY)
        loop.atomic_checkpoint(self.store / "ratings/evaluator.pt", state)
        rate_saved.save_champions(self.store, registry)
        self.evaluate(resume=True)
        self.assertEqual(len(self.jobs.calls), calls)
        self.assertEqual(before, {path: path.read_bytes() for path in paths})
        registry = rate_saved.load_champions(self.store)
        self.assertIsNone(registry["pending"])
        self.assertEqual(set(registry["skipped"]), set(self.ids))
        self.assertEqual(registry["evaluated"], {})

    def test_append_family_preserves_champion_and_overrides_only_its_cadence(self):
        # Simulate an existing evaluator which knows only the first saved model.
        (self.store / "ratings").mkdir(exist_ok=True)
        rate_saved.POLICY.update(evaluation_every=5, challenge_every=5)
        state = rate_saved.initialize(
            self.store, rate_saved.parse_args(["--model-store", str(self.store), "--family", "test1"])
        )
        state["candidates"] = self.ids[:1]
        state["models"] = {self.ids[0]: state["models"][self.ids[0]]}
        state["checkpoints"] = {self.ids[0]: state["checkpoints"][self.ids[0]]}
        registry = rate_saved.load_champions(self.store)
        registry["champion"] = self.ids[0]
        registry["history"].append({"model": self.ids[0]})
        registry["evaluated"][self.ids[0]] = {"complete": True, "decision": "promoted"}
        history = list(registry["history"])
        runner = rate_saved.ArtifactRunner(self.store, state, None)
        rate_saved.include_family(runner, registry, "test1", 1)
        self.assertEqual(registry["champion"], self.ids[0])
        self.assertEqual(registry["history"], history)
        self.assertEqual(state["candidates"], self.ids)
        self.assertEqual(state["policy"]["evaluation_every"], 5)
        self.assertEqual(state["family_evaluation_every"], {"test1": 1})
        changes = len(state["policy_changes"])
        rate_saved.include_family(runner, registry, "test1", 1)
        self.assertEqual(len(state["policy_changes"]), changes)
        rate_saved.evaluate(runner, registry)
        self.assertEqual(registry["champion"], self.ids[-1])
        self.assertFalse(registry.get("skipped"))

    def test_reports_have_no_redundant_threshold(self):
        self.evaluate()
        state = self.saved_state()
        self.assertNotIn("acceptance_rate", state["config"])
        for job in state["matches"].values():
            for key in ("gate", "reference"):
                if job.get(key):
                    self.assertNotIn("threshold", job[key]["report"])
                    self.assertNotIn("decision", job[key]["report"])
        legacy = EvaluationConfig.from_config({"acceptance_rate": 0.5})
        self.assertFalse(hasattr(legacy, "acceptance_rate"))

    def test_thresholds(self):
        decide = rate_saved.challenge_decision
        self.assertEqual(decide({"games": 10, "wins": 6, "draws": 4}), "screen-failed")
        self.assertEqual(decide({"games": 10, "wins": 7}), "continue")
        self.assertEqual(
            decide({"games": 10, "wins": 7}, {"games": 30, "wins": 20}), "challenge-failed"
        )
        self.assertEqual(decide({"games": 10, "wins": 7}, {"games": 30, "wins": 21}), "promoted")

    def test_small_metadata_and_single_model_copy(self):
        metadata = saved_models.verify(self.store / "models/test1/r0001")
        self.assertEqual(set(metadata), {"family", "round", "network_name", "progress", "source"})
        self.assertEqual(metadata["source"], {"run": "campaign", "iteration": 1})
        original = sorted((self.store / "models").rglob("checkpoint.pt"))
        self.evaluate()
        self.assertEqual(sorted(self.store.rglob("checkpoint.pt")), original)
        state = self.saved_state()
        self.assertEqual(state["models"][self.ids[0]]["directory"], "models/test1/r0001")

    def test_random_rooted_automatic_ladder_and_sgfs_only_rating_rebuild(self):
        before = (self.campaign / "loop.pt").read_bytes()
        result = self.evaluate()
        self.assertEqual(result["models"], 3)
        registry = rate_saved.load_champions(self.store)
        self.assertEqual(registry["champion"], self.ids[-1])
        self.assertEqual([entry["model"] for entry in registry["history"]], ["random", *self.ids])
        state = self.saved_state()
        matches = list(state["matches"].values())
        self.assertEqual(len([job for job in matches if job["kind"] == "reference"]), 3)
        self.assertEqual(len([job for job in matches if job["kind"] == "champion-screen"]), 2)
        self.assertEqual(len([job for job in matches if job["kind"] == "history"]), 1)
        self.assertEqual(state["rating_protocol"]["search"]["simulations"], 128)
        sgfs = list((self.store / "sgfs").glob("*/*.sgf"))
        self.assertEqual(len(sgfs), 150)
        self.assertTrue(all(rating_store.GAME_NAME.fullmatch(path.name) for path in sgfs))
        self.assertEqual(len(list((self.store / "sgfs").glob("*/*.nn.html"))), 150)
        summary = (self.store / "ratings/networks.csv").read_bytes()
        rating_store.refresh(
            self.store, EvaluationConfig.from_config(state["config"]), visualize=False
        )
        self.assertEqual((self.store / "ratings/networks.csv").read_bytes(), summary)
        self.assertFalse((self.store / "ratings/matches.csv").exists())
        self.assertEqual((self.campaign / "loop.pt").read_bytes(), before)
        calls = len(self.jobs.calls)
        self.evaluate(resume=True)
        self.assertEqual(len(self.jobs.calls), calls)

    def test_neural_challengers_rejected_without_random_games(self):
        self.jobs.wins["champion-screen"] = 6
        self.evaluate()
        state = self.saved_state()
        references = [j for j in state["matches"].values() if j["kind"] == "reference"]
        self.assertEqual([j["candidate_name"] for j in references], [self.ids[0]])
        registry = rate_saved.load_champions(self.store)
        self.assertEqual(registry["champion"], self.ids[0])
        for name in self.ids[1:]:
            self.assertEqual(registry["evaluated"][name]["decision"], "screen-failed")
        calls = len(self.jobs.calls)
        self.evaluate(resume=True)
        self.assertEqual(len(self.jobs.calls), calls)

    def test_neural_reference_runs_only_after_promotion_commit(self):
        play = rate_saved.play_match
        calls = []

        def checked(runner, registry, candidate, opponent, kind, games):
            if kind == "reference" and candidate != self.ids[0]:
                committed = rate_saved.load_champions(self.store)
                self.assertEqual(committed["champion"], candidate)
                self.assertEqual(committed["evaluated"][candidate]["decision"], "promoted")
            calls.append((candidate, kind))
            return play(runner, registry, candidate, opponent, kind, games)

        with patch.object(rate_saved, "play_match", checked):
            self.evaluate()
        for name in self.ids[1:]:
            kinds = [kind for candidate, kind in calls if candidate == name]
            self.assertLess(kinds.index("champion-extension"), kinds.index("reference"))

    def test_interrupted_post_promotion_reference_preserves_champion(self):
        play = rate_saved.play_match

        def interrupt(runner, registry, candidate, opponent, kind, games):
            if candidate == self.ids[1] and kind == "reference":
                self.jobs.interrupt_kind = "reference"
            return play(runner, registry, candidate, opponent, kind, games)

        with patch.object(rate_saved, "play_match", interrupt), self.assertRaises(KeyboardInterrupt):
            self.evaluate()
        registry = rate_saved.load_champions(self.store)
        self.assertEqual(registry["champion"], self.ids[1])
        self.assertFalse(registry["evaluated"][self.ids[1]]["complete"])
        before = {p: p.read_bytes() for p in (self.store / "sgfs").glob("*/*.sgf")}
        self.evaluate(resume=True)
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertEqual(rate_saved.load_champions(self.store)["champion"], self.ids[-1])
        references = [j for j in self.saved_state()["matches"].values()
                      if j["kind"] == "reference" and j["candidate_name"] == self.ids[1]]
        self.assertEqual(len(references), 1)
        self.assertEqual(len(references[0]["reference"]["attempts"]), 2)

    def test_rejection_keeps_champion_random(self):
        self.jobs.wins["reference"] = 6
        self.evaluate()
        self.assertEqual(rate_saved.load_champions(self.store)["champion"], "random")
        self.assertEqual(len(self.saved_state()["matches"]), 3)

    def test_27_out_of_40_does_not_promote(self):
        self.jobs.wins["champion-extension"] = 20
        self.evaluate()
        self.assertEqual(rate_saved.load_champions(self.store)["champion"], "random")

    def test_partial_pair_resumes_without_duplicate_games_or_retraining(self):
        self.jobs.interrupt_kind = "champion-extension"
        with self.assertRaises(KeyboardInterrupt):
            self.evaluate()
        paths = list((self.store / "sgfs").glob("*/*.sgf"))
        before = {path: path.read_bytes() for path in paths}
        training = len(self.stage_calls("continugo_train.train"))
        self.evaluate(resume=True)
        self.assertEqual(before, {path: path.read_bytes() for path in before})
        self.assertEqual(len(self.stage_calls("continugo_train.train")), training)
        self.assertEqual(len(list((self.store / "sgfs").glob("*/*.sgf"))), 150)
        self.assertEqual(rate_saved.load_champions(self.store)["champion"], self.ids[-1])

    def test_promotion_commit_is_idempotent(self):
        save = rate_saved.save_champions

        def interrupt(store, registry):
            save(store, registry)
            if registry["champion"] == self.ids[0]:
                raise KeyboardInterrupt

        with (
            patch.object(rate_saved, "save_champions", interrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.evaluate()
        self.evaluate(resume=True)
        self.assertEqual(len(rate_saved.load_champions(self.store)["history"]), 4)
