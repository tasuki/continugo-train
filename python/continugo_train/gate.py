"""Restartable experimental matches. Passing is evidence, never promotion permission."""

from .features import inspect_game
from .ratings import REFERENCE_ID
from .runtime import inference_environment
from .sampling import weighted_sizes


def plan(config, iteration, index, games=None):
    ordinal = (iteration - 1) * (config.gate_games if games is None else games) + index
    cycle = weighted_sizes(config.sizes, config.size_weights)
    return ordinal, cycle[(ordinal // 2) % len(cycle)], config.seed + ordinal // 2, ordinal % 2 == 0


def filename(ordinal, size, seed, candidate_black):
    color = "black" if candidate_black else "white"
    return f"game-{ordinal + 1:06d}-size{size}-seed{seed}-candidate-{color}.sgf"


def summarize(records, threshold):
    totals = {"wins": 0, "losses": 0, "draws": 0, "truncated": 0}
    per_size = {}
    for record in records:
        bucket = per_size.setdefault(record["board_size"], dict.fromkeys(totals, 0))
        if record["result"] is None:
            key = "truncated"
        elif record["result"] == "0":
            key = "draws"
        elif (record["result"].startswith("B+")) == record["candidate_black"]:
            key = "wins"
        else:
            key = "losses"
        totals[key] += 1
        bucket[key] += 1
    completed = len(records) - totals["truncated"]
    points = totals["wins"] + 0.5 * totals["draws"]
    score = points / completed if completed else None
    # Unknown games are not draws. Decide only when the planned match score is
    # guaranteed even if every truncated game resolves against that decision.
    planned = len(records)
    lower = points / planned if planned else 0.0
    upper = (points + totals["truncated"]) / planned if planned else 1.0
    if planned and points + 1e-10 >= threshold * planned:
        decision = "passed"
    elif planned and points + totals["truncated"] + 1e-10 < threshold * planned:
        decision = "failed"
    else:
        decision = "inconclusive"
    return {
        **totals,
        "completed": completed,
        "games": len(records),
        "score": score,
        "planned_score_bounds": (lower, upper),
        "threshold": threshold,
        "per_size": per_size,
        "decision": decision,
    }


def evaluate(runner, iteration, *, reference=False, adopt_only=False):
    config = runner.config
    key = "reference" if reference else "gate"
    count = config.reference_games if reference else config.gate_games
    max_moves = config.reference_max_moves if reference else config.gate_max_moves
    label = "REFERENCE" if reference else "EXPERIMENTAL MATCH"
    if iteration.get(key) is None:
        iteration[key] = {"attempts": [], "games": [None] * count, "report": None}
        runner.save()
    gate = iteration[key]
    if gate["report"] is not None:
        return gate["report"]
    if count == 0:
        gate["report"] = {"decision": "disabled", "disabled": True, "draws": 0, "truncated": 0}
        runner.save()
        return gate["report"]
    search = dict(runner.state["rating_protocol"]["search"])
    candidate = iteration["model"]
    incumbent = (
        {"checkpoint_sha256": REFERENCE_ID, "network_name": REFERENCE_ID}
        if reference
        else iteration["parent_model"]
    )
    density = runner.state["feature_spec"]["samples_per_diameter"]

    def adopt():
        for attempt in gate["attempts"]:
            if attempt.get("adopted"):
                continue
            for index in attempt["indices"]:
                if gate["games"][index]:
                    continue
                ordinal, size, seed, candidate_black = plan(
                    config, iteration["number"], index, count
                )
                directory = runner.path(attempt["path"])
                path = directory / filename(ordinal, size, seed, candidate_black)
                preferred = None
                if iteration.get("shared_output"):
                    black, white = (
                        (iteration["candidate_name"], iteration["opponent_name"])
                        if candidate_black
                        else (iteration["opponent_name"], iteration["candidate_name"])
                    )
                    preferred = directory / f"{black}-vs-{white}-seed{seed}.sgf"
                    if preferred.exists():
                        path = preferred
                if not path.exists():
                    if attempt["status"] == "succeeded":
                        raise ValueError(f"successful arena omitted {path}")
                    continue
                try:
                    info = inspect_game(path, density, config.feature_binary)
                except ValueError:
                    if attempt["status"] == "succeeded":
                        raise
                    runner.log(f"WARNING: retaining invalid partial arena SGF: {path}")
                    if iteration.get("shared_output"):
                        partial = path.with_suffix(f".incomplete-{len(gate['attempts'])}")
                        path.rename(partial)
                    continue
                black = candidate if candidate_black else incumbent
                white = incumbent if candidate_black else candidate
                root = info.get("root_comment")
                if (
                    not isinstance(root, dict)
                    or root.get("model") != black["checkpoint_sha256"]
                    or root.get("opponent") != white["checkpoint_sha256"]
                    or root.get("evaluation") != runner.state["rating_protocol"]["arena_sha256"]
                    or info.get("black_player")
                    != (
                        REFERENCE_ID
                        if black is incumbent and reference
                        else f"neural-{black['network_name']}"
                    )
                    or info.get("white_player")
                    != (
                        REFERENCE_ID
                        if white is incumbent and reference
                        else f"neural-{white['network_name']}"
                    )
                    or root.get("seed") != seed
                    or root.get("search") != search
                    or info["board_size"] != size
                    or not 0 < info["moves"] <= max_moves
                    or info.get("policy_visit_totals")
                    != [
                        None
                        if reference and ((ply % 2 == 0) != candidate_black)
                        else config.gate_simulations
                        for ply in range(info["moves"])
                    ]
                    or info["completed"] != info["labeled"]
                    or not info["completed"]
                    and info["moves"] != max_moves
                ):
                    raise ValueError(
                        f"arena record does not match its frozen two-model job: {path}"
                    )
                if preferred is not None and path != preferred:
                    if preferred.exists():
                        raise ValueError(f"refusing to overwrite a rated game: {preferred}")
                    path.rename(preferred)
                    path = preferred
                record = {
                    **runner.record(path),
                    "board_size": size,
                    "seed": seed,
                    "candidate_black": candidate_black,
                    "result": info["result"],
                    "moves": info["moves"],
                }
                gate["games"][index] = record
                if info["result"] == "0":
                    runner.log(f"WARNING DRAW in {label.lower()} test: {path}")
                if not info["labeled"]:
                    runner.log(f"WARNING TRUNCATED {label.lower()} game (not a draw): {path}")
                runner.save()
            attempt["adopted"] = True
            runner.save()

    adopt()
    if adopt_only:
        return None
    missing = [i for i, record in enumerate(gate["games"]) if record is None]
    groups = []
    for index in missing:
        if not groups or index != groups[-1][-1] + 1:
            groups.append([])
        groups[-1].append(index)
    for indices in groups:
        base = iteration.get("output_directory") or (
            "initial-reference"
            if iteration.get("initial")
            else f"iteration-{iteration['number']:04d}/{key}"
        )
        separator = "-" if iteration.get("flat_attempts") else "/"
        relative = (
            base
            if iteration.get("shared_output")
            else f"{base}{separator}attempt-{len(gate['attempts']) + 1:04d}"
        )
        directory = runner.path(relative)
        directory.mkdir(parents=True, exist_ok=bool(iteration.get("shared_output")))
        attempt = {"path": relative, "indices": indices, "status": "running", "adopted": False}
        gate["attempts"].append(attempt)
        runner.save()
        ordinal, _, _, _ = plan(config, iteration["number"], indices[0], count)
        library, environment = inference_environment(config.device, config.ort_library)
        command = [
            config.arena_binary,
            "--candidate",
            str(runner.check_model(candidate)),
            "--output",
            str(directory),
            "--games",
            str(len(indices)),
            "--game-offset",
            str(ordinal),
            "--seed",
            str(config.seed),
            "--sizes",
            ",".join(map(str, config.sizes)),
            "--size-weights",
            config.size_weights,
            "--max-moves",
            str(max_moves),
            "--device",
            config.device,
            "--ort-library",
            str(library),
        ]
        command += (
            ["--reference-random"]
            if reference
            else ["--incumbent", str(runner.check_model(incumbent))]
        )
        settings = {
            **search,
            **{
                name: getattr(config, name)
                for name in [
                    "inference_threads",
                    "feature_threads",
                    "cache_positions",
                ]
            },
        }
        for name, value in settings.items():
            command += ["--" + name.replace("_", "-"), str(value)]
        runner.log(
            f"iteration {iteration['number']}: {label.lower()} match, {len(indices)} missing games"
        )
        try:
            logfile = (
                runner.root / "ratings/evaluator.log"
                if iteration.get("shared_output")
                else directory / "arena.log"
            )
            runner.execute(command, logfile, environment)
        except BaseException:
            attempt["status"] = "interrupted"
            runner.save()
            if iteration.get("shared_output"):
                adopt()
            raise
        attempt["status"] = "succeeded"
        runner.save()
        adopt()
    gate["report"] = summarize(gate["games"], config.acceptance_rate)
    if reference:
        gate["report"].pop("decision")
        gate["report"].pop("threshold")
    runner.save()
    runner.log(f"{label} iteration={iteration['number']}: {gate['report']}")
    return gate["report"]
