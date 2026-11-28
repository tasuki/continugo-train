"""Playing defaults owned by Python; native workers require explicit values."""

from dataclasses import dataclass

from .sampling import DEFAULT_SIZE_WEIGHTS, DEFAULT_SIZES


@dataclass(frozen=True)
class SelfplayConfig:
    simulations: int = 100
    cpuct: float = 1.5
    widening: float = 3.0
    widening_exponent: float = 0.5
    proposal_attempts: int = 128
    independent_exploration: float = 0.5
    score_weight: float = 0.5
    useless_move_penalty_points: float = 0.5
    chosen_move_temperature_early: float = 0.75
    chosen_move_temperature: float = 0.15
    chosen_move_temperature_halflife: float = 19.0
    max_moves: int = 1024
    komi: float = 7.0
    komi_stddev: float = 2.0
    policy_simulations: int = 500
    policy_search_probability: float = 0.25
    concurrent_games: int = 32
    inference_batch_size: int = 32
    inference_batch_wait_ms: int = 2
    inference_threads: int = 1
    feature_threads: int = 6
    cache_positions: int = 64

    def search(self):
        return {name: getattr(self, name) for name in (
            "simulations", "cpuct", "widening", "widening_exponent", "proposal_attempts",
            "independent_exploration", "score_weight", "useless_move_penalty_points", "chosen_move_temperature_early",
            "chosen_move_temperature", "chosen_move_temperature_halflife",
        )}

    def inference(self):
        return {name: getattr(self, name) for name in (
            "concurrent_games", "inference_batch_size", "inference_batch_wait_ms",
            "inference_threads", "feature_threads", "cache_positions",
        )}

    def native_settings(self, *, neural=True):
        settings = {name: getattr(self, name) for name in SelfplayConfig.__dataclass_fields__}
        settings.update(sizes=",".join(map(str, DEFAULT_SIZES)), size_weights=DEFAULT_SIZE_WEIGHTS)
        if not neural:
            settings["concurrent_games"] = 1
        return settings

    def evaluation_search(self):
        return {k: v for k, v in SelfplayConfig.search(self).items()
                if not k.startswith("chosen_move_")}


def diagnostic_settings():
    config = SelfplayConfig()
    return {k: v for k, v in config.evaluation_search().items() if k != "simulations"} | {
        "threads": config.inference_threads,
        "feature_threads": config.feature_threads,
        "cache_positions": config.cache_positions,
    }


def arguments(settings):
    return [arg for key, value in settings.items() for arg in ("--" + key.replace("_", "-"), str(value))]
