//! Explicit test fixtures, not production defaults.
#![allow(dead_code)]
use continugo_train::search::SearchConfig;

#[rustfmt::skip]
pub fn search_config() -> SearchConfig {
    SearchConfig {
        simulations: 32, cpuct: 1.5, widening: 2.0, widening_exponent: 0.5,
        proposal_attempts: 128, independent_exploration: 0.25, score_weight: 0.5,
        useless_move_penalty_points: 0.0,
        chosen_move_temperature_early: 0.75, chosen_move_temperature: 0.15,
        chosen_move_temperature_halflife: 19.0, bootstrap_avoid_eyes: false, rollout: None,
    }
}

#[rustfmt::skip]
pub fn arguments(arena: bool) -> Vec<&'static str> {
    let mut args = vec![
        "--sizes", "9,13,19", "--size-weights", "9:1,13:0,19:0",
        "--max-moves", "512", "--simulations", "32", "--cpuct", "1.5",
        "--widening", "3", "--widening-exponent", "0.5", "--proposal-attempts", "128",
        "--independent-exploration", "0.5", "--score-weight", "0.5",
        "--inference-threads", "1", "--concurrent-games", "1",
        "--inference-batch-size", "8", "--inference-batch-wait-ms", "2",
        "--feature-threads", "6", "--cache-positions", "64",
    ];
    if arena {
        args.extend(["--games", "20"]);
    } else {
        args.extend([
            "--komi", "7", "--komi-stddev", "2", "--policy-simulations", "256",
            "--policy-search-probability", "0", "--chosen-move-temperature-early", "0.75",
            "--chosen-move-temperature", "0.15", "--chosen-move-temperature-halflife", "19",
        ]);
    }
    args
}
