import copy
import io
import json
import shutil
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
                if "RE[B+169.0]" not in text:
                    continue
                ordinal = int(path.name.split("-")[1]) - 1
                candidate_black = "candidate-black" in path.name
                black_wins = candidate_black == (ordinal < self.wins[kind])
                header = text.split(";B[", 1)[0].replace(
                    "RE[B+169.0]", "RE[B+169.0]" if black_wins else "RE[W+169.0]"
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
        self.stack.enter_context(patch.dict(rate_saved.POLICY, {"evaluation_every": 1}))
        self.stack.enter_context(patch.object(loop, "run_process", self.jobs))

        def report(_sgf, output, *_args, **_kwargs):
            Path(output).write_text("neural report")

        self.stack.enter_context(
            patch.object(rating_store.viewer_jobs, "worker_count", return_value=1)
        )
        self.stack.enter_context(
            patch.object(rating_store.viewer_jobs.nn_view, "generate_rated", report)
        )
        self.stack.enter_context(patch.object(rate_saved.plot_ratings, "update"))
        self.stack.enter_context(redirect_stdout(io.StringIO()))

    def evaluate(self, resume=False):
        args = ["--model-store", str(self.store)]
        args += ["--resume"] if resume else ["--family", "test1"]
        return rate_saved.run(rate_saved.parse_args(args))

    def saved_state(self):
        return torch.load(self.store / "ratings/evaluator.pt", weights_only=True)

    def test_runtime_upgrade_preserves_champion_history_and_settings(self):
        self.evaluate()
        state = self.saved_state()
        registry_path = self.store / "champions.json"
        registry = json.loads(registry_path.read_text())
        before_registry = copy.deepcopy(registry)
        before_state = copy.deepcopy(state)
        directory = self.root / "numbered-schema-runtime"
        directory.mkdir()
        for key, name in (
            ("selfplay_binary", "continugo-train"),
            ("feature_binary", "features"),
            ("arena_binary", "arena"),
            ("rating_binary", "rating"),
        ):
            shutil.copy2(state["config"][key], directory / name)
        with (directory / "arena").open("ab") as stream:
            stream.write(b"\nnumbered-schema-runtime-test\n")  # genuinely different executable hash
        runner = rate_saved.ArtifactRunner(self.store, state, 0)
        rate_saved.upgrade_runtime(runner, registry, directory)
        runner.verify_runtime()
        self.assertEqual(registry["champion"], before_registry["champion"])
        self.assertEqual(registry["history"], before_registry["history"])
        self.assertEqual(registry["evaluated"], before_registry["evaluated"])
        self.assertEqual(state["matches"], before_state["matches"])
        self.assertEqual(state["trials"], before_state["trials"])
        self.assertEqual(
            state["rating_protocol"]["search"], before_state["rating_protocol"]["search"]
        )
        self.assertTrue((Path(state["runtime_changes"][-1]["backup"]) / "evaluator.pt").is_file())
        self.assertEqual(len(state["runtime_changes"]), 1)
        cache, _ = rating_store.refresh(self.store, runner.config, visualize=False)
        self.assertEqual(len(cache["protocols"]), 2)
        self.assertEqual(len(cache["records"]), len(before_state["rating_cache"]["records"]))
        sgf = next((self.store / "sgfs").glob("*/*.sgf"))
        original = sgf.read_text()
        old_hash = before_state["rating_protocol"]["arena_sha256"]
        self.assertIn(old_hash, original)
        sgf.write_text(original.replace(old_hash, "0" * 64))
        with self.assertRaisesRegex(ValueError, "unregistered evaluation runtime"):
            rating_store.refresh(self.store, runner.config, visualize=False)
        sgf.write_text(original)
        rate_saved.upgrade_runtime(runner, registry, directory)
        self.assertEqual(len(state["runtime_changes"]), 1)
        registry["pending"] = {"candidate": self.ids[0]}
        rate_saved.upgrade_runtime(runner, registry, directory)  # identical runtime is resumable
        other = self.root / "different-runtime"
        shutil.copytree(directory, other)
        with self.assertRaisesRegex(ValueError, "pending"):
            rate_saved.upgrade_runtime(runner, registry, other)

    def test_joint_fit_cache_and_legacy_cache_match_full_rebuild(self):
        self.evaluate()
        config = EvaluationConfig.from_config(self.saved_state()["config"])
        cache, expected = rating_store.refresh(self.store, config, visualize=False)
        # Cached states are presentation only, never the input to the joint fit.
        cache["states"][self.ids[0]]["rating"] = -99999
        _, cached = rating_store.refresh(self.store, config, cache, visualize=False)
        self.assertEqual(cached, expected)
        legacy = {
            "protocol": cache["protocol"],
            "groups": cache["groups"],
            "states": cache["states"],
        }
        _, rebuilt = rating_store.refresh(self.store, config, legacy, visualize=False)
        self.assertEqual(rebuilt, expected)

    def test_only_every_fifth_round_gets_any_evaluation(self):
        rate_saved.POLICY.update(evaluation_every=5)
        self.run_loop(self.campaign, resume=True, extra=["--iterations", "5"])
        self.evaluate()
        registry = rate_saved.load_champions(self.store)
        self.assertEqual(registry["champion"], "test1-r0005")
        self.assertEqual(
            [entry["model"] for entry in registry["history"]], ["random", "test1-r0005"]
        )
        for index in range(1, 5):
            name = f"test1-r{index:04d}"
            self.assertNotIn("skipped", registry)
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
        rate_saved.POLICY.update(evaluation_every=5)
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
        self.assertNotIn("skipped", registry)
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
        rate_saved.POLICY.update(evaluation_every=5)
        state = self.saved_state()
        registry = json.loads((self.store / "champions.json").read_text())
        state["policy"] = dict(rate_saved.POLICY)
        registry["policy"] = dict(rate_saved.POLICY)
        loop.atomic_checkpoint(self.store / "ratings/evaluator.pt", state)
        rate_saved.save_champions(self.store, registry)
        self.evaluate(resume=True)
        self.assertEqual(len(self.jobs.calls), calls)
        self.assertEqual(before, {path: path.read_bytes() for path in paths})
        registry = rate_saved.load_champions(self.store)
        self.assertIsNone(registry["pending"])
        self.assertNotIn("skipped", registry)
        self.assertEqual(registry["evaluated"], {})

    def test_append_family_preserves_champion_and_overrides_only_its_cadence(self):
        # Simulate an existing evaluator which knows only the first saved model.
        (self.store / "ratings").mkdir(exist_ok=True)
        rate_saved.POLICY.update(evaluation_every=5)
        state = rate_saved.initialize(
            self.store,
            rate_saved.parse_args(["--model-store", str(self.store), "--family", "test1"]),
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

    def test_compact_registry_preserves_evidence_and_full_checkpoint_audit(self):
        registry = rate_saved.load_champions(self.store)
        policy_change = {"family": "test1", "added_models": self.ids, "evaluation_every": 1}
        runtime_change = {
            "previous_config": {"device": "cuda"},
            "executables": {"arena": "large payload"},
            "previous_protocol": {"arena_sha256": "old"},
            "protocol": {"arena_sha256": "new"},
        }
        registry.update(
            skipped={self.ids[0]: {"reason": "off-cycle"}},
            policy_changes=[policy_change],
            runtime_changes=[runtime_change],
            family_evaluation_every={"test1": 1},
        )
        rate_saved.save_champions(self.store, registry)
        loaded = rate_saved.load_champions(self.store)
        self.assertNotIn("skipped", loaded)
        for field in ("policy_changes", "runtime_changes", "family_evaluation_every"):
            self.assertNotIn(field, loaded)
        self.assertIn("added_models", policy_change)
        self.assertIn("executables", runtime_change)

    def test_csv_preserves_rows_and_appends_new_networks(self):
        import csv

        self.evaluate()
        path = self.store / "networks.csv"
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        # An existing order is durable, even when it differs from lexical order.
        reordered = [rows[0], rows[2], rows[1]]
        rate_saved.ratings._atomic_csv(path, rating_store.COLUMNS, reordered)
        config = EvaluationConfig.from_config(self.saved_state()["config"])
        rating_store.refresh(self.store, config, visualize=False)
        with path.open(newline="") as stream:
            names = [row["network"] for row in csv.DictReader(stream)]
        self.assertEqual(names, [row["network"] for row in reordered] + [self.ids[-1]])
        path.unlink()
        rating_store.refresh(self.store, config, visualize=False)
        with path.open(newline="") as stream:
            self.assertEqual(
                [row["network"] for row in csv.DictReader(stream)], ["random", *self.ids]
            )

    def test_thresholds(self):
        decide = rate_saved.challenge_decision
        self.assertEqual(decide({"games": 10, "wins": 6, "draws": 4}), "screen-failed")
        self.assertEqual(decide({"games": 10, "wins": 7}), "continue")
        self.assertEqual(
            decide({"games": 10, "wins": 7}, {"games": 30, "wins": 20}), "challenge-failed"
        )
        self.assertEqual(decide({"games": 10, "wins": 7}, {"games": 30, "wins": 21}), "promoted")
        self.assertNotIn("challenge_every", rate_saved.POLICY)
        self.assertNotIn("reference_games", rate_saved.POLICY)
        with patch.dict(
            rate_saved.POLICY,
            {"screen_games": 2, "win_rate_required": 0.5, "extension_games": 4},
        ):
            self.assertEqual(decide({"games": 2, "wins": 1}), "continue")
            self.assertEqual(decide({"games": 2, "wins": 1}, {"games": 4, "wins": 2}), "promoted")
            self.assertEqual(
                decide({"games": 2, "wins": 1}, {"games": 4, "wins": 1}), "challenge-failed"
            )

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
        self.assertEqual(len([job for job in matches if job["kind"] == "reference"]), 1)
        self.assertEqual(len([job for job in matches if job["kind"] == "champion-screen"]), 2)
        self.assertEqual(len([job for job in matches if job["kind"] == "history"]), 6)
        self.assertTrue(all(job["games"] == 50 for job in matches if job["kind"] == "history"))
        for name, defeated in zip(self.ids, ["random", *self.ids[:-1]]):
            opponents = state["trials"][name]["history_opponents"]
            self.assertIn(defeated, opponents)
            self.assertLessEqual(len(opponents), 3)
            self.assertEqual(len(opponents), len(set(opponents)))
        self.assertEqual(len(state["trials"][self.ids[-1]]["history_opponents"]), 3)
        self.assertEqual(state["rating_protocol"]["search"]["simulations"], 128)
        sgfs = list((self.store / "sgfs").glob("*/*.sgf"))
        self.assertEqual(len(sgfs), 420)
        self.assertTrue(all(rating_store.GAME_NAME.fullmatch(path.name) for path in sgfs))
        self.assertEqual(len(list((self.store / "sgfs").glob("*/*.nn.html"))), 420)
        summary = (self.store / "networks.csv").read_bytes()
        rating_store.refresh(
            self.store, EvaluationConfig.from_config(state["config"]), visualize=False
        )
        self.assertEqual((self.store / "networks.csv").read_bytes(), summary)
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

    def test_history_runs_only_after_promotion_commit(self):
        play = rate_saved.play_match
        calls = []

        def checked(runner, registry, candidate, opponent, kind, games):
            if kind == "history":
                committed = rate_saved.load_champions(self.store)
                self.assertEqual(committed["champion"], candidate)
                self.assertEqual(committed["evaluated"][candidate]["decision"], "promoted")
            calls.append((candidate, kind))
            return play(runner, registry, candidate, opponent, kind, games)

        with patch.object(rate_saved, "play_match", checked):
            self.evaluate()
        for name in self.ids[1:]:
            kinds = [kind for candidate, kind in calls if candidate == name]
            self.assertLess(kinds.index("champion-extension"), kinds.index("history"))
            self.assertNotIn("reference", kinds)

    def test_interrupted_history_random_match_preserves_champion(self):
        play = rate_saved.play_match

        def interrupt(runner, registry, candidate, opponent, kind, games):
            if candidate == self.ids[1] and kind == "history" and opponent == "random":
                self.jobs.interrupt_kind = "reference"
            return play(runner, registry, candidate, opponent, kind, games)

        with (
            patch.object(rate_saved, "play_match", interrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.evaluate()
        registry = rate_saved.load_champions(self.store)
        self.assertEqual(registry["champion"], self.ids[1])
        self.assertFalse(registry["evaluated"][self.ids[1]]["complete"])
        before = {p: p.read_bytes() for p in (self.store / "sgfs").glob("*/*.sgf")}
        self.evaluate(resume=True)
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertEqual(rate_saved.load_champions(self.store)["champion"], self.ids[-1])
        references = [
            j
            for j in self.saved_state()["matches"].values()
            if j["kind"] == "history"
            and j["opponent_name"] == "random"
            and j["candidate_name"] == self.ids[1]
        ]
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
        self.assertEqual(len(list((self.store / "sgfs").glob("*/*.sgf"))), 420)
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
