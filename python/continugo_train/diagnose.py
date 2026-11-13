"""Stage-balanced held-out outcome, score and ownership diagnostics (no training)."""

import argparse
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path

import torch

from .data import inspect_corpus
from .features import FeatureClient, ownership_weights
from .model import NetworkConfig, PolicyValueNet
from .train import checkpoint_splits, feature_spec


def summarize(rows):
    result = {"positions": len(rows)}
    for name in (
        "value_mse",
        "score_mae",
        "score_mse",
        "score_bias",
        "ownership_mse",
        "score_ownership_gap",
    ):
        result[name] = sum(row[name] for row in rows) / len(rows)
    result["score_rmse"] = result["score_mse"] ** 0.5
    for name in ("value", "score"):
        mean = sum(row[name] for row in rows) / len(rows)
        result[name + "_std"] = (sum((row[name] - mean) ** 2 for row in rows) / len(rows)) ** 0.5
    bins = defaultdict(list)
    for row in rows:
        bins[min(9, int((row["value"] + 1) * 5))].append(row)
    result["outcome_calibration"] = [
        {
            "bin": b,
            "positions": len(group),
            "predicted_points": sum((r["value"] + 1) / 2 for r in group) / len(group),
            "observed_points": sum((r.get("outcome_target", r["outcome"]) + 1) / 2 for r in group)
            / len(group),
        }
        for b, group in sorted(bins.items())
    ]
    bins.clear()
    for row in rows:
        bins[min(9, int((row["score"] + 1) * 5))].append(row)
    result["score_calibration"] = [
        {
            "bin": b,
            "positions": len(group),
            "predicted": sum(r["score"] for r in group) / len(group),
            "observed": sum(r["score_target"] for r in group) / len(group),
        }
        for b, group in sorted(bins.items())
    ]
    return result


def diagnose(checkpoint, data, *, device="cpu", max_games=32, binary=None):
    raw = Path(checkpoint).read_bytes()
    saved = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    density = saved["feature_spec"]["samples_per_diameter"]
    if saved["feature_spec"] != feature_spec(density):
        raise ValueError("incompatible checkpoint; use a new score/ownership model")
    history = checkpoint_splits(saved)
    games, _ = inspect_corpus(data, density, binary)
    games = sorted(
        (g for g in games if history.get(g.game_id) == "validation"), key=lambda g: g.game_id
    )[:max_games]
    if not games:
        raise ValueError(
            "no ancestral held-out games found; training games are not diagnostic holdouts"
        )
    model = PolicyValueNet(NetworkConfig(**saved["config"])).to(device).eval()
    model.load_state_dict(saved["model"])
    rows = []
    with FeatureClient(binary) as client, torch.inference_mode():
        for game in games:
            # Eligible pre-move positions, not terminal boards; no data resplitting.
            plies = game.plies
            choices = sorted({0, len(plies) // 2, max(0, len(plies) - 2), len(plies) - 1})
            for index in choices:
                ply = plies[index]
                sample = client.load(game.path, ply, density=density, file_sha256=game.file_sha256)
                if sample.outcome is None or sample.score is None or sample.ownership is None:
                    raise ValueError("missing final-game targets")
                _, _, value, score, owned = model.encode_training(sample.coarse[None].to(device))
                truth = sample.ownership.to(device)
                weights = ownership_weights(game.resolution, device=device)
                v, s = value.item(), score.item()
                target = (
                    (-1.0 if sample.outcome == 0 else sample.outcome)
                    if saved.get("outcome_semantics") == "jigo-loss-v1"
                    else sample.outcome
                )
                stage = "opening" if index == 0 else "late" if index >= len(plies) - 2 else "middle"
                rows.append(
                    {
                        "game_id": game.game_id,
                        "sgf_sha256": game.file_sha256,
                        "ply": ply,
                        "board_size": game.board_size,
                        "stage": stage,
                        "value": v,
                        "outcome": sample.outcome,
                        "outcome_target": target,
                        "score": s,
                        "score_target": sample.score,
                        "value_mse": (v - target) ** 2,
                        "score_mae": abs(s - sample.score),
                        "score_mse": (s - sample.score) ** 2,
                        "score_bias": s - sample.score,
                        "ownership_mse": ((owned[0] - truth).square() * weights).sum().item(),
                        "score_ownership_gap": abs(s - (owned[0] * weights).sum().item()),
                    }
                )
    by_group = defaultdict(list)
    for row in rows:
        by_group[f"size{row['board_size']}/{row['stage']}"].append(row)
    return {
        "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
        "summary": summarize(rows),
        "by_size_stage": {key: summarize(group) for key, group in sorted(by_group.items())},
        "positions": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-games", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--feature-binary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.max_games <= 0 or args.threads <= 0:
        parser.error("max-games and threads must be positive")
    torch.set_num_threads(args.threads)
    result = diagnose(
        args.checkpoint,
        args.data,
        device=args.device,
        max_games=args.max_games,
        binary=args.feature_binary,
    )
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x") as stream:
            stream.write(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
