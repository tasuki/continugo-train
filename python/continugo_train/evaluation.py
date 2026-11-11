"""Standalone rating configuration and runner, independent of training scheduling."""

from dataclasses import dataclass, fields

from . import ratings
from .loop import ROOT, LoopConfig, LoopRunner


@dataclass(frozen=True)
class EvaluationConfig(LoopConfig):
    arena_binary: str = str(ROOT / "target/release/arena")
    rating_binary: str = str(ROOT / "target/release/rating")
    reference_games: int = 10
    reference_max_moves: int = 2048
    gate_games: int = 20
    gate_simulations: int = 128
    gate_max_moves: int = 512

    @classmethod
    def from_config(cls, config):
        """Accept both current training settings and historical evaluation settings."""
        names = {field.name for field in fields(cls)}
        return cls(**{name: value for name, value in config.items() if name in names})

    def validate(self):
        super().validate()
        if any(not 0 <= n < 2**32 or n % 2 for n in [self.gate_games, self.reference_games]):
            raise ValueError("gate-games/reference-games must be even nonnegative u32 counts")
        if (
            not 0 < self.gate_simulations < 2**32
            or min(self.gate_max_moves, self.reference_max_moves) <= 0
        ):
            raise ValueError("positive evaluation budgets required")


def validate_output(root, store):
    """Keep evaluator work away from publications and the authoritative ledger."""
    root, store = root.resolve(), store.resolve()
    jobs = store / "ratings" / "jobs"
    if store.is_relative_to(root) or root.is_relative_to(store) and not root.is_relative_to(jobs):
        raise ValueError(
            "evaluation output must be outside the store or under models/ratings/jobs/"
        )


def source_protocol(source, config):
    """Freeze evaluation settings here; training no longer owns a rating protocol."""
    previous = source.get("rating_protocol")
    legacy = previous["version"] == 1 if previous else source.get("version") == 5
    settings = ratings.protocol(config, legacy=legacy)
    if previous is not None and settings != previous:
        raise ValueError("source evaluation binaries/settings changed")
    return settings


class EvaluationRunner(LoopRunner):
    def __init__(self, root, state, lock_fd):
        self.root, self.state, self.lock_fd = root, state, lock_fd
        self.config = EvaluationConfig.from_config(state["config"])

    def verify_runtime(self):
        super().verify_runtime()
        if (
            ratings.protocol(self.config, legacy=self.state["rating_protocol"]["version"] == 1)
            != self.state["rating_protocol"]
        ):
            raise ValueError("evaluation/rating executable changed; use the frozen runtime")

    def rate_evaluation(self, job, key, checkpoints):
        match = job[key]
        if not match["games"] or match.get("rating_event"):
            return
        result = ratings.ingest(
            self.config.model_store,
            self.state["rating_protocol"],
            [self.check(record) for record in match["games"]],
            checkpoints=[self.check(record) for record in checkpoints],
            kind=key,
            decision=match["report"].get("decision"),
            rating_binary=self.config.rating_binary,
            feature_binary=self.config.feature_binary,
        )
        match["rating_event"] = result
        self.save()
        self.log(f"RATINGS {key}: {result}")
