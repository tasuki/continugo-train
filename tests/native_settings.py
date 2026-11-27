"""Native integration-test calls with explicit caller-owned playing settings."""

import subprocess
from pathlib import Path

from continugo_train.evaluation import EvaluationConfig
from continugo_train.selfplay import SelfplayConfig, arguments, diagnostic_settings


def run(command, **kwargs):
    binary = Path(command[0]).name
    config = SelfplayConfig(policy_search_probability=0.0)  # Small fixed-budget fixtures.
    settings = (config.native_settings(neural="neural" in command) if binary == "continugo-train"
                else EvaluationConfig().arena_settings() if binary == "arena"
                else diagnostic_settings() if binary == "infer" else {})
    return subprocess.run([command[0], *arguments(settings), *command[1:]],
                          check=kwargs.pop("check", False), **kwargs)
