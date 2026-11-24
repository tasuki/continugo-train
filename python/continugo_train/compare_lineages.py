"""Prepare or resume frozen, rated champion comparisons across two lineages."""

import argparse
import fcntl
import json
import math
import re
import shutil
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, replace
from pathlib import Path

import torch

from . import archive, gate, lineage_ratings, plot_ratings, ratings, viewer_jobs
from .evaluation import EvaluationConfig, EvaluationRunner
from .loop import LoopRunner
from .paths import utc_now
from .rate_saved import champion_lock, model_references
from .runtime import inference_environment
from .saved_models import saved_sources
from .train import atomic_checkpoint

VERSION = "lineage-comparison-v1"


@contextmanager
def comparison_lock(root):
    directory = root / "ratings/comparisons"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another lineage comparison is running") from error
        yield stream.fileno()


def latest_champions(store, count):
    history = json.loads((store / "champions.json").read_text())["history"]
    names = [entry["model"] for entry in history if entry["model"] != "random"]
    if count <= 0 or len(names) < count:
        raise ValueError(f"{store}: fewer than {count} neural champions")
    return names[-count:]


class ComparisonRunner(EvaluationRunner):
    def __init__(self, root, state, lock_fd, state_path):
        super().__init__(root, state, lock_fd)
        self.state_path = state_path

    def verify_runtime(self):
        LoopRunner.verify_runtime(self)
        expected = {**ratings.protocol(self.config), "komi": self.config.komi}
        if expected != self.state["rating_protocol"]:
            raise ValueError("comparison runtime/settings changed")

    def save(self):
        atomic_checkpoint(self.state_path, self.state)

    def log(self, text):
        print(text, flush=True)
        with self.state_path.with_suffix(".log").open("a") as stream:
            stream.write(text + "\n")


def historical_inspectors(root, catalog, current, overrides=()):
    """Find a rules inspector that preserves each lineage's recorded scores."""
    from .champion_wins import root_properties

    explicit = {}
    for value in overrides:
        lineage, separator, path = value.partition("=")
        if not separator or not path:
            raise ValueError("historical inspector must be LINEAGE=PATH")
        explicit[lineage] = Path(path).resolve()
    chosen = {}
    for registry in sorted(root.glob("*/champions.json")):
        lineage = registry.parent.name
        samples = {}
        for path in sorted((registry.parent / "sgfs").glob("*/*.sgf")):
            if path.name.startswith("game-"):
                continue
            metadata = json.loads(root_properties(path)["C"][0])
            samples.setdefault(metadata["evaluation"], path)
        candidates = [Path(current)]
        for source in catalog.values():
            context = source.get("source_context")
            if source["lineage"] == lineage and context:
                candidates.append(Path(context["config"]["feature_binary"]))
        if lineage in explicit:
            candidates = [explicit[lineage]]
        for binary in dict.fromkeys(candidates):
            if not binary.is_file():
                continue
            try:
                for path in samples.values():
                    ratings.inspect_game(path, binary=binary)
            except (ValueError, OSError):
                continue
            chosen[lineage] = binary.resolve()
            break
        else:
            raise ValueError(
                f"no compatible historical inspector for {lineage}; "
                "supply --historical-inspector LINEAGE=PATH"
            )
    return chosen


def prepare(root, args, state_path):
    if state_path.exists():
        raise ValueError("comparison already exists; use run --name to resume")
    if args.lineage_a == args.lineage_b:
        raise ValueError("choose two different lineages")
    if args.games <= 0 or args.games % 2 or not math.isfinite(args.komi):
        raise ValueError("games must be positive and even; komi must be finite")
    stores = [root / name for name in (args.lineage_a, args.lineage_b)]
    selected = [latest_champions(store, args.latest) for store in stores]
    catalog = lineage_ratings.sources(root)
    for names, store in zip(selected, stores, strict=True):
        if any(name not in catalog or catalog[name].get("pruned") for name in names):
            raise ValueError(f"selected champion artifacts unavailable: {store}")
    # Borrow only search/runtime settings, never the candidate queue or promotion policy.
    source_path = stores[0] / "ratings/evaluator.pt"
    if not source_path.exists():
        raise ValueError("lineage-a must have evaluator.pt to supply frozen search settings")
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    base = EvaluationConfig.from_config(source["config"])
    native = args.runtime.resolve()
    for binary in ("continugo-train", "features", "arena", "rating"):
        if not (native / binary).is_file():
            raise ValueError(f"missing comparison executable: {native / binary}")
    # Verify the supplied arena actually supports fixed-komi matches before freezing it.
    import subprocess

    help_text = subprocess.run(
        [str(native / "arena"), "--help"], capture_output=True, text=True, check=True
    ).stdout
    if "--komi" not in help_text or "--check-models" not in help_text:
        raise ValueError("rebuild arena: runtime needs --komi and --check-models")
    inspectors = historical_inspectors(
        root, catalog, native / "features", args.historical_inspector
    )
    runtime = root / "ratings/comparisons" / f"{args.name}-runtime"
    if runtime.exists():
        raise ValueError(f"orphan runtime directory; inspect before retrying: {runtime}")
    runtime.mkdir()
    for binary in ("continugo-train", "features", "arena", "rating"):
        shutil.copy2(native / binary, runtime / binary)
        archive.seal_file(runtime / binary)
        (runtime / binary).chmod(0o555)
    frozen_inspectors = {}
    for lineage, binary in inspectors.items():
        target = runtime / f"historical-{lineage}-features"
        shutil.copy2(binary, target)
        archive.seal_file(target)
        target.chmod(0o555)
        frozen_inspectors[lineage] = {"path": str(target), "sha256": archive.digest(target)}
    config = replace(
        base,
        model_store=str(root),
        selfplay_binary=str(runtime / "continugo-train"),
        feature_binary=str(runtime / "features"),
        arena_binary=str(runtime / "arena"),
        rating_binary=str(runtime / "rating"),
        gate_games=args.games,
        device=args.device or base.device,
        komi=args.komi,
        komi_stddev=0.0,
    )
    config.validate()
    library, environment = inference_environment(config.device, config.ort_library)
    models, checkpoints = {}, {}
    for store, names in zip(stores, selected, strict=True):
        selected_sources = {key: s for key, s in saved_sources(store).items() if s["id"] in names}
        # References are relative to the shared artifacts root; models are not copied.
        m, c = model_references(root, selected_sources)
        models.update(m)
        checkpoints.update(c)
    from .nn_view import load_checkpoint

    for name in models:
        # Load-only preflight: --check-models exits before creating output or
        # generating any games. Check HTML support too, not just an empty report.
        subprocess.run(
            [
                config.arena_binary,
                "--candidate",
                str(root / models[name]["directory"]),
                "--reference-random",
                "--check-models",
                "--komi",
                str(args.komi),
                "--device",
                config.device,
                "--ort-library",
                str(library),
                "--output",
                str(runtime / "unused-check-output"),
            ],
            env=environment,
            check=True,
        )
        loaded = load_checkpoint(root / checkpoints[name]["path"])
        del loaded
    seed_path = root / "ratings/comparisons/next-seed.json"
    seed = json.loads(seed_path.read_text()) if seed_path.exists() else 98000000
    end = seed + len(selected[0]) * len(selected[1]) * args.games // 2
    if end >= 2**64:
        raise ValueError("comparison seed overflow")
    from .loop import atomic_bytes

    atomic_bytes(seed_path, (json.dumps(end) + "\n").encode())
    settings = {**ratings.protocol(config), "komi": args.komi}
    jobs = []
    for candidate in selected[0]:
        for opponent in selected[1]:
            jobs.append(
                {
                    "number": 1,
                    "candidate_name": candidate,
                    "opponent_name": opponent,
                    "model": models[candidate],
                    "parent_model": models[opponent],
                    "checkpoint": checkpoints[candidate],
                    "parent_actor_checkpoint": checkpoints[opponent],
                    "seed": seed,
                    "games": args.games,
                    "komi": args.komi,
                    "shared_output": True,
                    "gate": None,
                    "output_directory": None,
                }
            )
            seed += args.games // 2
    state = {
        "version": VERSION,
        "root": str(root),
        "lineages": [s.name for s in stores],
        "selected": selected,
        "config": asdict(config),
        "rating_protocol": settings,
        "feature_spec": source["feature_spec"],
        "models": models,
        "checkpoints": checkpoints,
        "runtime_fingerprints": {
            "selfplay": archive.digest(config.selfplay_binary),
            "features": archive.digest(config.feature_binary),
        },
        "historical_inspectors": frozen_inspectors,
        "prepared_at": utc_now().isoformat(),
        "jobs": jobs,
        "complete": False,
    }
    atomic_checkpoint(state_path, state)
    return state


def run(root, state, state_path, lock_fd):
    if state.get("version") != VERSION or state.get("root") != str(root):
        raise ValueError("incompatible or relocated comparison manifest")
    runner = ComparisonRunner(root, state, lock_fd, state_path)
    # EvaluationRunner verifies the base protocol; komi is an explicit extension.
    expected = {**ratings.protocol(runner.config), "komi": runner.config.komi}
    if expected != state["rating_protocol"]:
        raise ValueError("comparison runtime/settings changed")
    for name in state["models"]:
        runner.check(state["checkpoints"][name])
        runner.check_model(state["models"][name])
    checkpoints = {
        record["sha256"]: runner.check(record) for record in state["checkpoints"].values()
    }
    for number, job in enumerate(state["jobs"], 1):
        runner.config = replace(EvaluationConfig.from_config(state["config"]), seed=job["seed"])
        if job["output_directory"] is None:
            job["output_directory"] = f"sgfs/{utc_now().date().isoformat()}"
            runner.save()
        runner.log(
            f"COMPARISON {number}/{len(state['jobs'])}: "
            f"{job['candidate_name']} vs {job['opponent_name']}, "
            f"{job['games']} games, komi {job['komi']}"
        )
        gate.evaluate(runner, job)
        reports = []
        for record in job["gate"]["games"]:
            path = runner.check(record)
            output = path.with_suffix(".nn.html")
            if not output.exists():
                info = ratings.inspect_game(path, binary=runner.config.feature_binary)
                ratings._inspect(
                    path,
                    state["rating_protocol"],
                    {
                        checksum: {f"neural-{state['models'][name]['network_name']}"}
                        for name, checksum in (
                            (n, state["checkpoints"][n]["sha256"]) for n in state["models"]
                        )
                    },
                    runner.config.feature_binary,
                    info=info,
                )
                reports.append((path, output, info, checkpoints, runner.config.feature_binary))
        viewer_jobs.generate(reports)
        lineage_ratings.refresh(root, runner.config.feature_binary)
        plot_ratings.update(root / state["lineages"][0])
        job["rated"] = True
        runner.save()
    state["complete"] = True
    runner.save()
    runner.log("COMPARISON COMPLETE")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    parser.add_argument("--name", required=True)
    parser.add_argument("--lineage-a")
    parser.add_argument("--lineage-b")
    parser.add_argument("--latest", type=int)
    parser.add_argument("--games", type=int)
    parser.add_argument("--komi", type=float)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument(
        "--historical-inspector",
        action="append",
        default=[],
        metavar="LINEAGE=PATH",
        help="override automatic historical rules inspection",
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.name):
        parser.error("name must contain only letters, digits, underscores and hyphens")
    if args.action == "run":
        if args.historical_inspector or any(
            getattr(args, key) is not None
            for key in ("lineage_a", "lineage_b", "latest", "games", "komi", "runtime", "device")
        ):
            parser.error("run restores frozen settings; supply only --name and --artifacts-root")
    else:
        args.latest = args.latest if args.latest is not None else 3
        args.games = args.games if args.games is not None else 40
        args.komi = args.komi if args.komi is not None else 0.0
        args.runtime = args.runtime or Path("target/release")
    root = args.artifacts_root.resolve()
    state_path = root / "ratings/comparisons" / f"{args.name}.pt"
    if args.action == "prepare" and any(
        not name or not re.fullmatch(r"[A-Za-z0-9_-]+", name)
        for name in (args.lineage_a, args.lineage_b)
    ):
        parser.error("prepare requires --lineage-a and --lineage-b")
    with comparison_lock(root) as lock_fd, ExitStack() as stack:
        if args.action == "prepare":
            lineages = [args.lineage_a, args.lineage_b]
        else:
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            lineages = state["lineages"]
        for lineage in sorted(lineages):
            stack.enter_context(champion_lock(root / lineage))
        if args.action == "prepare":
            state = prepare(root, args, state_path)
            print(
                f"Prepared {len(state['jobs'])} pairings, "
                f"{sum(j['games'] for j in state['jobs'])} games. No games started."
            )
            print(f"Manifest: {state_path}")
            for names in state["selected"]:
                print(", ".join(names))
        else:
            run(root, state, state_path, lock_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
