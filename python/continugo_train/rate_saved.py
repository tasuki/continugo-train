"""One shared evaluator, one model copy, date-grouped SGFs as the game records."""

import argparse
import fcntl
import json
import os
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

import torch

from . import archive, gate, plot_ratings, rating_store, ratings
from .evaluation import EvaluationConfig, EvaluationRunner
from .loop import atomic_bytes
from .paths import ROOT, utc_now
from .saved_models import saved_sources
from .train import atomic_checkpoint

POLICY = {
    "evaluation_every": 5,
    "challenge_every": 5,
    "screen_games": 10,
    "screen_wins": 7,
    "extension_games": 30,
    "total_wins": 28,
    "reference_games": 10,
    "history_opponents": 2,
    "history_games": 10,
}


def challenge_decision(screen, extension=None):
    if screen["games"] != 10:
        raise ValueError("screen must contain ten games")
    if screen["wins"] < 7:
        return "screen-failed"
    if extension is None:
        return "continue"
    if extension["games"] != 30:
        raise ValueError("extension must contain thirty games")
    return "promoted" if screen["wins"] + extension["wins"] >= 28 else "challenge-failed"


@contextmanager
def champion_lock(store):
    directory = store / "ratings"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".evaluator.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another evaluator is running") from error
        yield stream.fileno()


def load_champions(store):
    path = store / "ratings/champions.json"
    if path.is_symlink():
        raise ValueError("champion registry cannot be a symlink")
    if path.exists():
        value = json.loads(path.read_text())
        if value.get("policy") != POLICY or "champion" not in value:
            raise ValueError("incompatible champion policy")
        return value
    return {
        "policy": dict(POLICY),
        "champion": "random",
        "history": [{"model": "random"}],
        "evaluated": {},
        "pending": None,
        "next_seed": 97000000,
    }


def save_champions(store, registry):
    path = store / "ratings/champions.json"
    atomic_bytes(path, (json.dumps(registry, indent=2) + "\n").encode())
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    archive.sync_directory(path.parent)


class ArtifactRunner(EvaluationRunner):
    def save(self):
        atomic_checkpoint(self.root / "ratings/evaluator.pt", self.state)

    def log(self, text):
        print(text, flush=True)
        with (self.root / "ratings/evaluator.log").open("a") as stream:
            stream.write(text + "\n")


def model_references(store, sources):
    models, checkpoints = {}, {}
    for source in sources.values():
        name = source["id"]

        def record(key, source=source):
            path = Path(source[key]["path"])
            if path.is_symlink() or not path.resolve().is_relative_to(store):
                raise ValueError("saved models must live directly inside the artifact store")
            return {"path": str(path.relative_to(store)), "sha256": source[key]["sha256"]}

        checkpoints[name] = record("checkpoint")
        models[name] = {
            "directory": str(Path(source["checkpoint"]["path"]).parent.relative_to(store)),
            "checkpoint_sha256": source["checkpoint_sha256"],
            "network_name": source["network_name"],
            "encoder": record("encoder"),
            "policy": record("policy"),
        }
    return models, checkpoints


def initialize(store, args):
    sources = {
        key: value
        for key, value in saved_sources(store).items()
        if value["family"] == (args.family or "cg5")
    }
    if not sources:
        raise ValueError("no saved models for this family")
    pool = sorted(sources.values(), key=lambda value: value["round"])
    source = pool[-1]["source_context"]
    if source is None:
        raise ValueError(
            "training runtime not found; keep the source run when creating an evaluator"
        )
    base = EvaluationConfig.from_config(source["config"])
    native = Path(base.selfplay_binary).parent
    config = replace(
        base,
        model_store=str(store),
        gate_games=10,
        reference_games=10,
        gate_simulations=args.simulations if args.simulations is not None else 128,
        gate_max_moves=1024,
        reference_max_moves=2048,
        device=args.device or "cuda",
        seed=args.seed if args.seed is not None else 97000000,
        arena_binary=source["config"].get("arena_binary", str(native / "arena")),
        rating_binary=source["config"].get("rating_binary", str(native / "rating")),
    )
    config.validate()
    models, checkpoints = model_references(store, sources)
    state = {
        "root": str(store),
        "family": args.family or "cg5",
        "policy": dict(POLICY),
        "config": asdict(config),
        "feature_spec": source["feature_spec"],
        "rating_protocol": ratings.protocol(config),
        "runtime_fingerprints": source["runtime_fingerprints"],
        "models": models,
        "checkpoints": checkpoints,
        "candidates": [value["id"] for value in pool],
        "matches": {},
        "trials": {},
        "rating_cache": {},
    }
    atomic_checkpoint(store / "ratings/evaluator.pt", state)
    return state


def play_match(runner, registry, candidate, opponent, kind, games):
    state = runner.state
    key = f"{candidate}:{kind}:{opponent}"
    neural = opponent != "random"
    if key not in state["matches"]:
        seed = registry["next_seed"]
        if seed + games // 2 > 2**64:
            raise ValueError("evaluation seeds overflow u64")
        registry["next_seed"] += games // 2
        save_champions(runner.root, registry)
        state["matches"][key] = {
            "number": 1,
            "output_directory": f"sgfs/{utc_now().date().isoformat()}",
            "shared_output": True,
            "candidate_name": candidate,
            "opponent_name": opponent,
            "model": state["models"][candidate],
            "checkpoint": state["checkpoints"][candidate],
            "seed": seed,
            "games": games,
            "kind": kind,
            "opponent": opponent,
            "gate": None,
            "reference": None,
        }
        if neural:
            state["matches"][key].update(
                parent_model=state["models"][opponent],
                parent_actor_checkpoint=state["checkpoints"][opponent],
            )
        runner.save()
    job = state["matches"][key]
    match_key = "gate" if neural else "reference"
    runner.check(job["checkpoint"])
    runner.check_model(job["model"])
    if neural:
        runner.check(job["parent_actor_checkpoint"])
        runner.check_model(job["parent_model"])
    for record in (job[match_key] or {}).get("games", []):
        if record:
            runner.check(record)
    original = runner.config
    runner.config = replace(original, seed=job["seed"], gate_games=games, reference_games=games)
    try:
        gate.evaluate(runner, job, reference=not neural)
        match = job[match_key]
        if not match.get("rated"):
            state["rating_cache"], _summary = rating_store.refresh(
                runner.root, runner.config, state.get("rating_cache")
            )
            match["rated"] = True
            runner.save()
        return match["report"]
    finally:
        runner.config = original


def evaluate(runner, registry):
    state = runner.state
    if registry["champion"] != "random" and registry["champion"] not in state["models"]:
        raise ValueError("current champion model unavailable")
    skipped = registry.setdefault("skipped", {})
    for number, candidate in enumerate(state["candidates"], 1):
        if registry["evaluated"].get(candidate, {}).get("complete") or candidate in skipped:
            continue
        if registry["pending"] not in {None, candidate}:
            raise ValueError("resume the pending candidate before changing the queue")
        family, round_text = candidate.rsplit("-r", 1)
        round_number = int(round_text)
        cadence = state.get("family_evaluation_every", {}).get(family)
        evaluation_every = cadence or state["policy"]["evaluation_every"]
        challenge_every = cadence or state["policy"]["challenge_every"]
        if round_number % evaluation_every != 0:
            skipped[candidate] = {"reason": "off-cycle", "recorded_at": utc_now().isoformat()}
            registry["pending"] = None
            state.setdefault("skipped", []).append(candidate)
            runner.save()
            save_champions(runner.root, registry)
            runner.log(
                f"Skipped {candidate}: all evaluation is restricted to multiples of {evaluation_every}"
            )
            continue
        registry["pending"] = candidate
        save_champions(runner.root, registry)
        if candidate not in state["trials"]:
            state["trials"][candidate] = {
                "opponent": registry["champion"],
                "decision": None,
                "history_opponents": None,
            }
            runner.save()
        trial = state["trials"][candidate]
        opponent = trial["opponent"]
        if "challenge_due" not in trial:
            # Finish a previously started challenge, but never start an off-cycle one.
            started = bool(trial.get("screen")) or any(
                job["candidate_name"] == candidate and job["kind"].startswith("champion-")
                for job in state["matches"].values()
            )
            trial["challenge_due"] = (
                started
                or round_number % challenge_every == 0
            )
            runner.save()
        runner.log(
            f"Candidate {number}/{len(state['candidates'])}: {candidate} vs champion {opponent}"
        )
        if trial["decision"] is None and not trial["challenge_due"]:
            trial["decision"] = "not-challenged"
            runner.save()
        if trial["decision"] is None:
            screen = (
                play_match(runner, registry, candidate, "random", "reference", 10)
                if opponent == "random"
                else play_match(runner, registry, candidate, opponent, "champion-screen", 10)
            )
            trial["screen"] = screen
            decision = challenge_decision(screen)
            if decision == "continue":
                extension = play_match(
                    runner, registry, candidate, opponent, "champion-extension", 30
                )
                trial["extension"] = extension
                decision = challenge_decision(screen, extension)
            trial["decision"] = decision
            runner.save()
        if candidate not in registry["evaluated"]:
            if registry["champion"] != opponent:
                raise ValueError("champion changed during the frozen challenge")
            receipt = {
                "decision": trial["decision"],
                "opponent": opponent,
                "screen_wins": trial.get("screen", {}).get("wins"),
                "total_wins": (
                    trial["screen"]["wins"] + (trial.get("extension") or {}).get("wins", 0)
                )
                if trial.get("screen")
                else None,
                "complete": False,
                "recorded_at": utc_now().isoformat(),
            }
            registry["evaluated"][candidate] = receipt
            if trial["decision"] == "promoted":
                registry["champion"] = candidate
                registry["history"].append(
                    {
                        "model": candidate,
                        "opponent": opponent,
                        "recorded_at": receipt["recorded_at"],
                    }
                )
            save_champions(runner.root, registry)
            plot_ratings.update(runner.root)
        # Calibration is a reward for promotion, never a prerequisite. Commit
        # the champion first; a resumed partial reference batch cannot undo it.
        # During initial bootstrap the random challenge already supplies these
        # ten games, so do not schedule a duplicate reference match.
        if trial["decision"] == "promoted" and opponent != "random":
            play_match(runner, registry, candidate, "random", "reference", 10)
        if trial["history_opponents"] is None:
            opponents = []
            if trial["decision"] == "promoted":
                states = state["rating_cache"]["states"]
                target = states[candidate]["rating"]
                opponents = sorted(
                    [
                        name
                        for name in registry["evaluated"]
                        if name in state["models"] and name not in {candidate, opponent}
                    ],
                    key=lambda name: (abs(states[name]["rating"] - target), name),
                )[:2]
            trial["history_opponents"] = opponents
            runner.save()
        for historical in trial["history_opponents"]:
            play_match(runner, registry, candidate, historical, "history", 10)
        registry["evaluated"][candidate]["complete"] = True
        registry["pending"] = None
        save_champions(runner.root, registry)
        plot_ratings.update(runner.root)
        runner.log(f"Finished {candidate}: {trial['decision']}; champion={registry['champion']}")


def include_family(runner, registry, family, cadence=None):
    """Append compatible saved models without resetting the shared champion/history."""
    sources = {key: value for key, value in saved_sources(runner.root).items()
               if value["family"] == family}
    if not sources:
        raise ValueError(f"no saved models for family {family}")
    models, checkpoints = model_references(runner.root, sources)
    state = runner.state
    ordered = sorted(sources.values(), key=lambda source: source["round"])
    added = []
    for source in ordered:
        name = source["id"]
        if name in state["models"]:
            if state["models"][name] != models[name] or state["checkpoints"][name] != checkpoints[name]:
                raise ValueError(f"saved model identity changed: {name}")
        else:
            saved = torch.load(source["checkpoint"]["path"], map_location="cpu", weights_only=True)
            if saved["feature_spec"] != state["feature_spec"]:
                raise ValueError(f"incompatible feature semantics: {name}")
            added.append(name)
    old_cadence = state.get("family_evaluation_every", {}).get(family)
    if cadence is not None and cadence <= 0:
        raise ValueError("family evaluation cadence must be positive")
    state["models"].update(models)
    state["checkpoints"].update(checkpoints)
    for source in ordered:
        if source["id"] not in state["candidates"]:
            state["candidates"].append(source["id"])
    if cadence is not None:
        state.setdefault("family_evaluation_every", {})[family] = cadence
        registry.setdefault("family_evaluation_every", {})[family] = cadence
        # A cadence change can make previously skipped models eligible. Never
        # discard completed evidence or alter another family's cadence.
        for source in ordered:
            name = source["id"]
            if source["round"] % cadence == 0:
                registry.get("skipped", {}).pop(name, None)
                if name in state.get("skipped", []):
                    state["skipped"].remove(name)
    if added or cadence is not None and cadence != old_cadence:
        change = dict(family=family, added_models=added,
                      evaluation_every=cadence if cadence is not None else old_cadence,
                      recorded_at=utc_now().isoformat())
        state.setdefault("policy_changes", []).append(change)
        registry.setdefault("policy_changes", []).append(change)
    runner.save()
    save_champions(runner.root, registry)


def run(args):
    store = (args.model_store or ROOT / "artifacts").resolve()
    archive.initialize_store(store)
    with champion_lock(store) as lock:
        path = store / "ratings/evaluator.pt"
        if args.resume:
            if any(
                getattr(args, name) is not None
                for name in ["family", "simulations", "seed", "device"]
            ):
                raise ValueError("resume restores frozen settings")
            state = torch.load(path, map_location="cpu", weights_only=True)
            if state.get("root") != str(store) or state.get("policy") != POLICY:
                raise ValueError("incompatible or relocated evaluator checkpoint")
        else:
            if path.exists():
                raise ValueError("the shared evaluator exists; use --resume")
            state = initialize(store, args)
        # Ignore the removed generic report threshold in historical state.
        state["config"].pop("acceptance_rate", None)
        runner = ArtifactRunner(store, state, lock)
        runner.verify_runtime()
        registry = load_champions(store)
        if args.include_family:
            include_family(runner, registry, args.include_family, args.family_evaluation_every)
        if not registry["evaluated"] and not state["matches"]:
            registry["next_seed"] = state["config"]["seed"]
        evaluate(runner, registry)
        runner.log("Evaluation complete; training and publication unchanged")
        return {"models": len(state["candidates"]), "status": "complete"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-store", type=Path, help="artifact root (default artifacts/)")
    parser.add_argument("--family", help="model family to evaluate (default cg5)")
    parser.add_argument(
        "--simulations", type=int, help="fixed simulations per neural move (default 128)"
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--include-family", help="append saved models to the existing shared evaluator")
    parser.add_argument("--family-evaluation-every", type=int,
                        help="evaluation/challenge cadence for --include-family only (1 = every model)")
    args = parser.parse_args(argv)
    if args.family_evaluation_every is not None and (
        not args.include_family or args.family_evaluation_every <= 0
    ):
        parser.error("positive --family-evaluation-every requires --include-family")
    return args


def main(argv=None):
    try:
        run(parse_args(argv))
    except KeyboardInterrupt:
        raise SystemExit("Interrupted; resume with --resume") from None
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(f"rate-saved error: {error}") from error


if __name__ == "__main__":
    main()
