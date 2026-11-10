"""Uncertainty-driven rated matchmaking; no training, actor gating or publication."""

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, replace
from itertools import combinations
from pathlib import Path

import torch

from . import gate, ratings
from .evaluation import EvaluationConfig, EvaluationRunner, source_protocol, validate_output
from .features import inspect_game
from .loop import atomic_bytes, loop_lock
from .round_robin import available_models, snapshot_models
from .train import atomic_checkpoint

VERSION = "adaptive-ratings-v1"
Q = math.log(10) / 400


def information(player, opponent):
    """Expected per-game Glicko precision gain, not an outcome prediction oracle."""
    g = 1 / math.sqrt(1 + 3 * (Q * opponent["deviation"] / math.pi) ** 2)
    x = Q * g * (player["rating"] - opponent["rating"])
    p = 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))
    return (Q * g) ** 2 * p * (1 - p), p


def pair_statistics(evidence):
    pairs = {}
    colors = defaultdict(set)
    for game in evidence.values():
        a, b = sorted((game["black"], game["white"]))
        pair = pairs.setdefault((a, b), {"attempted": 0, "completed": 0, "pairs": 0})
        pair["attempted"] += 1
        if not game["result"]:
            continue
        pair["completed"] += 1
        colors[(a, b, game["seed"], game["board_size"])].add(game["black"])
    for (a, b, _seed, _size), seen in colors.items():
        if len(seen) == 2:
            pairs[(a, b)]["pairs"] += 1
    return pairs


def progress(targets, states, pairs, policy):
    result = {}
    for target in targets:
        opponent_pairs = {
            (b if a == target else a): stats["pairs"]
            for (a, b), stats in pairs.items()
            if target in (a, b) and stats["pairs"]
        }
        seeds = sum(opponent_pairs.values())
        opponents = sum(n >= policy["min_opponent_pairs"] for n in opponent_pairs.values())
        state = states[target]
        done = (
            bool(state["anchored"])
            and state["deviation"] <= policy["target_rd"]
            and seeds >= policy["min_pairs"]
            and opponents >= policy["min_opponents"]
        )
        result[target] = {
            "rating": state["rating"],
            "deviation": state["deviation"],
            "paired_seeds": seeds,
            "paired_games": 2 * seeds,
            "opponents": opponents,
            "opponent_pairs": opponent_pairs,
            "done": done,
        }
    return result


def select_match(states, targets, pairs, policy):
    """Maximize expected target variance reduction, subject to opponent diversity.

    Near-equal strengths give most Fisher information. Uncertain opponents provide
    less, and unfinished targets receive priority. Discount already-covered
    opponents until the minimum diversity requirement is met. Never play random.
    """
    status = progress(targets, states, pairs, policy)
    needy = {network for network, item in status.items() if not item["done"]}
    best = None
    for a, b in combinations(sorted(states), 2):
        if not needy.intersection((a, b)):
            continue
        stats = pairs.get((a, b), {"attempted": 0, "completed": 0, "pairs": 0})
        completion = (stats["completed"] + 2) / (stats["attempted"] + 2)
        score = 0.0
        for target, opponent in [(a, b), (b, a)]:
            if target not in needy:
                continue
            variance = states[target]["deviation"] ** 2
            gain, _ = information(states[target], states[opponent])
            reduction = variance - variance / (1 + policy["batch_games"] * variance * gain)
            if (
                status[target]["opponents"] < policy["min_opponents"]
                and stats["pairs"] >= policy["min_opponent_pairs"]
            ):
                reduction *= 0.01
            score += reduction / policy["target_rd"] ** 2
        score *= completion
        _, expected = information(states[a], states[b])
        choice = (score, a, b, expected)
        if best is None or choice[:3] > best[:3]:
            best = choice
    if best is None:
        return None
    score, a, b, expected = best
    return {
        "players": [a, b],
        "expected_score_first": expected,
        "information_score": score,
        "ratings_before": {a: states[a], b: states[b]},
    }


def refresh(runner):
    """Cache validated metadata once per new SGF; count complete color/seed pairs."""
    state = runner.state
    pool = state["models"]
    with ratings.transaction(runner.config.model_store) as (tables, _):
        states = {n: dict(tables.state(n)) for n in pool}
        rows = [
            r
            for r in tables.matches
            if r["black"] in pool
            and r["white"] in pool
            and r["black"] != r["white"]
            and ratings._protocol_from_row(r) == state["rating_protocol"]
        ]
    for row in rows:
        checksum = row["sgf_sha256"]
        if checksum in state["evidence"]:
            continue
        path = ratings.evidence_path(runner.config.model_store, row)
        info = inspect_game(
            path, state["feature_spec"]["samples_per_diameter"], runner.config.feature_binary
        )
        state["evidence"][checksum] = {
            "black": row["black"],
            "white": row["white"],
            "result": row["result"],
            "seed": info["root_comment"]["seed"],
            "board_size": info["board_size"],
        }
    pairs = pair_statistics(state["evidence"])
    state["progress"] = progress(state["targets"], states, pairs, state["policy"])
    runner.save()
    atomic_bytes(
        runner.root / "progress.json",
        (
            json.dumps(
                {
                    "policy": state["policy"],
                    "additional_games_played": len(state["jobs"]) * runner.config.gate_games,
                    "stop_reason": state.get("stop_reason"),
                    "targets": state["progress"],
                },
                indent=2,
            )
            + "\n"
        ).encode(),
    )
    for target, item in state["progress"].items():
        runner.log(
            f"PROGRESS {state['models'][target]['network_name']}: "
            f"{item['rating']:.1f} +/- {item['deviation']:.1f}; "
            f"{item['paired_seeds']} color pairs; {item['opponents']} covered opponents; "
            f"{'target met' if item['done'] else 'more evidence needed'}"
        )
    return states, pairs


def initialize(root, source, args):
    policy = {
        "target_rd": args.target_rd,
        "min_pairs": args.min_pairs,
        "min_opponents": args.min_opponents,
        "min_opponent_pairs": args.min_opponent_pairs,
        "batch_games": args.batch_games,
        "max_games": args.max_games,
    }
    if (
        not math.isfinite(args.target_rd)
        or args.target_rd <= 0
        or any(policy[k] <= 0 for k in ["min_pairs", "min_opponents", "min_opponent_pairs"])
        or args.batch_games <= 0
        or args.batch_games % 2
        or args.max_games <= 0
        or args.max_games % args.batch_games
    ):
        raise ValueError(
            "positive uncertainty/evidence limits, even batches and a divisible game budget required"
        )
    if args.latest <= 0 or args.pool_latest is not None and args.pool_latest < args.latest:
        raise ValueError("positive --latest, and --pool-latest must include all targets")
    config = replace(
        EvaluationConfig.from_config(source["config"]), gate_games=args.batch_games, seed=args.seed
    )
    config.validate()
    if args.seed + args.max_games // 2 > 2**64:
        raise ValueError("evaluation seeds overflow u64")
    store = Path(config.model_store).resolve()
    validate_output(root, store)
    settings = source_protocol(source, config)
    available = available_models(store, feature_spec=source["feature_spec"])
    rows = [row for row, _ in available]
    count = args.pool_latest if args.pool_latest is not None else len(rows)
    targets = (
        list(dict.fromkeys(args.targets))
        if args.targets
        else [r["network_id"] for r in rows[: args.latest]]
    )
    pool_ids = {r["network_id"] for r in rows[:count]}
    if count <= args.min_opponents or args.latest > len(rows) and not args.targets:
        raise ValueError("not enough neural opponents/targets for the confidence policy")
    if not targets or not set(targets) <= pool_ids:
        raise ValueError(
            "explicit targets must be registered neural checkpoints in the opponent pool"
        )
    bundles = [bundle for _, bundle in available[:count]]
    models, checkpoints = snapshot_models(root, source, bundles)
    return {
        "version": VERSION,
        "root": str(root),
        "config": asdict(config),
        "feature_spec": source["feature_spec"],
        "rating_protocol": settings,
        "runtime_fingerprints": source["runtime_fingerprints"],
        "models": {m["checkpoint_sha256"]: m for m in models},
        "checkpoints": {m["checkpoint_sha256"]: c for m, c in zip(models, checkpoints)},
        "targets": targets,
        "policy": policy,
        "jobs": [],
        "evidence": {},
        "stop_reason": None,
    }


def run(args, *, parent_lock_fd=None):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with loop_lock(root) as lock_fd:
        if args.resume:
            # Defaults apply only on creation, never silently override a saved policy.
            if args.loop or any(getattr(args, k) is not None for k in SETUP_DEFAULTS):
                raise ValueError(
                    "resume restores saved settings; supply only --output and --resume"
                )
            state = torch.load(root / "loop.pt", map_location="cpu", weights_only=True)
            if state.get("version") != VERSION or state.get("root") != str(root):
                raise ValueError("incompatible/relocated adaptive evaluation")
        else:
            if any(p.name != ".loop.lock" for p in root.iterdir()):
                raise ValueError("output is not empty; use --resume or a fresh directory")
            if args.loop is None:
                raise ValueError("--loop is required for frozen evaluation settings")
            for key, default in SETUP_DEFAULTS.items():
                if getattr(args, key) is None:
                    setattr(args, key, default)
            source = torch.load(args.loop / "loop.pt", map_location="cpu", weights_only=True)
            state = initialize(root, source, args)
            atomic_checkpoint(root / "loop.pt", state)
        parent_locks = (
            (parent_lock_fd,) if isinstance(parent_lock_fd, int) else tuple(parent_lock_fd or ())
        )
        locks = (*parent_locks, lock_fd) if parent_locks else lock_fd
        runner = EvaluationRunner(root, state, locks)
        runner.verify_runtime()
        for model in state["models"].values():
            runner.check_model(model)
        for checkpoint in state["checkpoints"].values():
            runner.check(checkpoint)
        while True:
            if state["jobs"] and not (state["jobs"][-1].get("gate") or {}).get("rating_event"):
                job = state["jobs"][-1]
                for record in (job.get("gate") or {}).get("games", []):
                    if record:
                        runner.check(record)
                gate.evaluate(runner, job)
                runner.rate_evaluation(
                    job, "gate", [job["checkpoint"], job["parent_actor_checkpoint"]]
                )
            states, pairs = refresh(runner)
            if all(item["done"] for item in state["progress"].values()):
                state["stop_reason"] = "confidence-target-met"
            elif len(state["jobs"]) * runner.config.gate_games >= state["policy"]["max_games"]:
                state["stop_reason"] = "game-budget-exhausted; confidence target NOT met"
            if state["stop_reason"]:
                runner.save()
                refresh(runner)
                runner.log(f"STOP: {state['stop_reason']}")
                return state
            selection = select_match(states, state["targets"], pairs, state["policy"])
            if selection is None:
                raise ValueError("no eligible neural matchup for unfinished targets")
            a, b = selection["players"]
            job = {
                "number": len(state["jobs"]) + 1,
                "model": state["models"][a],
                "parent_model": state["models"][b],
                "checkpoint": state["checkpoints"][a],
                "parent_actor_checkpoint": state["checkpoints"][b],
                "selection": selection,
                "gate": None,
            }
            state["jobs"].append(job)
            runner.save()
            runner.log(
                f"ADAPTIVE MATCH {job['number']}: {job['model']['network_name']} vs "
                f"{job['parent_model']['network_name']}; expected first score "
                f"{selection['expected_score_first']:.3f}; {runner.config.gate_games} games"
            )


SETUP_DEFAULTS = {
    "latest": 3,
    "targets": None,
    "pool_latest": None,
    "target_rd": 75.0,
    "min_pairs": 20,
    "min_opponents": 2,
    "min_opponent_pairs": 4,
    "batch_games": 4,
    "max_games": 360,
    "seed": 9300000,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--loop", type=Path, help="source loop directory, read-only")
    parser.add_argument("--resume", action="store_true")
    for key, default in SETUP_DEFAULTS.items():
        if key == "targets":
            parser.add_argument(
                "--targets", nargs="+", help="explicit full checkpoint IDs instead of --latest"
            )
            continue
        kind = float if key == "target_rd" else int
        parser.add_argument(
            "--" + key.replace("_", "-"),
            type=kind,
            help=f"new evaluation default: {default if default is not None else 'all registered neural networks'}",
        )
    return parser.parse_args(argv)


def main(argv=None):
    try:
        run(parse_args(argv))
    except KeyboardInterrupt:
        raise SystemExit("Interrupted; resume with --output SAME_DIRECTORY --resume") from None
    except (ValueError, OSError, RuntimeError) as error:
        raise SystemExit(f"adaptive ratings error: {error}") from error


if __name__ == "__main__":
    main()
