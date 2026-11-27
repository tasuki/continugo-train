// Parallel rules queries create many short-lived vectors. Keep their allocations
// thread-local rather than contending on the system allocator's shared arenas.
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

use clap::Parser;
use continugo_rules::{Board, territory};
use continugo_train::{
    feature_cache::FeatureCacheArgs,
    inference::{Evaluator, InferenceDevice, OnnxEvaluator, batching::BatchingConfig},
    nn_html::NnHtmlArgs,
    sampling::weighted_sizes,
    search::{SearchParameters, rollout::RolloutArgs},
    selfplay::{
        Engine, GameRecord, PolicySearchConfig, SelfplayConfig, generate, generate_neural,
        generate_profiled,
        parallel::{ParallelConfig, generate_parallel},
    },
};
use std::{
    error::Error,
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

#[derive(Parser)]
#[command(
    version,
    args_override_self = true,
    about = "ContinuGo random, unguided MCTS and native ONNX neural MCTS self-play"
)]
struct Args {
    #[arg(long, value_enum, default_value = "mcts")]
    engine: Engine,
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
    games: u32,
    /// Supported board equivalents; allocation follows --size-weights.
    #[arg(long, value_delimiter = ',', required = true, num_args = 1.., action = clap::ArgAction::Set)]
    sizes: Vec<i32>,
    /// Board mix supplied by the caller; 'uniform' cycles literal sizes.
    #[arg(long)]
    size_weights: String,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Mean komi, in White's point-equivalents.
    #[arg(long, allow_negative_numbers = true)]
    komi: f64,
    /// Game-level Gaussian variation, rounded to half points and capped at 0..14.
    #[arg(long)]
    komi_stddev: f64,
    #[arg(long)]
    max_moves: usize,
    #[arg(long)]
    simulations: u32,
    /// Total search budget on policy-training positions (with --policy-search-probability).
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    policy_simulations: u32,
    /// Chance of an expensive policy-training search; 0 keeps fixed-budget training.
    /// Randomization requires --simulations >= 32. Cheap moves keep value labels.
    #[arg(long)]
    policy_search_probability: f64,
    /// Initial self-play move-selection temperature (both search budgets).
    #[arg(long)]
    chosen_move_temperature_early: f64,
    /// Asymptotic late-game self-play temperature; 0 selects maximum visits.
    #[arg(long)]
    chosen_move_temperature: f64,
    /// Temperature decay half-life in moves on 19x19, scaled by board width.
    #[arg(long)]
    chosen_move_temperature_halflife: f64,
    #[command(flatten)]
    search: SearchParameters,
    /// Avoid noncapturing own-eye fills in bootstrap MCTS only (not neural play).
    #[arg(long)]
    bootstrap_avoid_eyes: bool,
    #[command(flatten)]
    rollout: RolloutArgs,
    /// ONNX model directory containing encoder.onnx and policy.onnx (neural engine).
    #[arg(long)]
    model: Option<PathBuf>,
    /// Native ONNX Runtime >=1.20 shared library; otherwise use ORT_DYLIB_PATH.
    #[arg(long)]
    ort_library: Option<PathBuf>,
    #[arg(long)]
    inference_threads: usize,
    /// Active independent games (1..=64).
    #[arg(long)]
    concurrent_games: usize,
    /// Maximum neural inference batch, bucketed by resolution (1..=32).
    #[arg(long)]
    inference_batch_size: usize,
    /// Maximum collection delay; partial batches always run (0..=1000 ms).
    #[arg(long)]
    inference_batch_wait_ms: u64,
    /// Bounded CPU pool for exact coordinate features (1 = serial).
    #[arg(long)]
    feature_threads: usize,
    /// Maximum cached coarse-network contexts; 0 disables, payload capped at 32 MiB.
    #[arg(long)]
    cache_positions: usize,
    #[arg(long, value_enum, default_value = "cuda")]
    device: InferenceDevice,
    /// Time random-engine stages and moves; embed diagnostics in SGF comments.
    #[arg(long)]
    profile: bool,
    /// Write machine-readable run totals and inference stages to a new JSON file.
    #[arg(long)]
    stats_json: Option<PathBuf>,
    #[command(flatten)]
    nn_view: NnHtmlArgs,
    #[command(flatten)]
    feature_cache: FeatureCacheArgs,
    /// Directory for self-contained .sgf records, or '-' for stdout.
    #[arg(long, default_value = "-")]
    output: PathBuf,
}

impl Args {
    fn parallel_config(&self) -> Result<ParallelConfig, String> {
        let neural = matches!(self.engine, Engine::Neural);
        let parallel = ParallelConfig {
            games: self.games,
            seed: self.seed,
            concurrent_games: self.concurrent_games,
            batching: BatchingConfig {
                max_batch: self.inference_batch_size,
                max_wait: Duration::from_millis(self.inference_batch_wait_ms),
            },
        };
        parallel.validate()?;
        if !neural && parallel.concurrent_games != 1 {
            return Err("concurrent games/inference batching require --engine neural".into());
        }
        Ok(parallel)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    args.nn_view.validate(false, false)?;
    if args.nn_view.nn_html
        && (!matches!(args.engine, Engine::Neural) || args.output.as_os_str() == "-")
    {
        return Err("--nn-html requires --engine neural and an --output directory".into());
    }
    if args.profile && !matches!(args.engine, Engine::Random) {
        return Err("--profile currently requires --engine random".into());
    }
    let parallel = args.parallel_config()?;
    if !args.policy_search_probability.is_finite()
        || !(0.0..=1.0).contains(&args.policy_search_probability)
    {
        return Err("--policy-search-probability must be finite and in 0..1".into());
    }
    let config = SelfplayConfig {
        engine: args.engine,
        komi: args.komi,
        komi_stddev: args.komi_stddev,
        max_moves: args.max_moves,
        policy_search: (args.policy_search_probability > 0.0).then_some(PolicySearchConfig {
            simulations: args.policy_simulations,
            probability: args.policy_search_probability,
        }),
        search: continugo_train::search::SearchConfig {
            chosen_move_temperature_early: args.chosen_move_temperature_early,
            chosen_move_temperature: args.chosen_move_temperature,
            chosen_move_temperature_halflife: args.chosen_move_temperature_halflife,
            bootstrap_avoid_eyes: args.bootstrap_avoid_eyes,
            rollout: args.rollout.config(),
            ..args.search.config(args.simulations)
        },
    };
    config.validate()?;
    let mut evaluator = match (args.engine, &args.model) {
        (Engine::Neural, Some(path)) => Some(OnnxEvaluator::load(
            path,
            args.ort_library.as_deref(),
            args.inference_threads,
            args.device,
        )?),
        (Engine::Neural, None) => return Err("--engine neural requires --model".into()),
        (_, Some(_)) => return Err("--model requires --engine neural".into()),
        _ => None,
    };
    if let Some(evaluator) = &mut evaluator {
        evaluator.configure_features(args.feature_threads, args.cache_positions)?;
        evaluator.profile_coarse = args.stats_json.is_some();
        eprintln!(
            "ONNX device={} model={} threads={} feature_threads={} cache_positions={} concurrent_games={} batch_size={} batch_wait_ms={} proposal_policy={} geometry_budget={:?}",
            evaluator.device.name(),
            evaluator.model_id(),
            args.inference_threads,
            args.feature_threads,
            args.cache_positions,
            parallel.concurrent_games,
            parallel.batching.max_batch,
            args.inference_batch_wait_ms,
            evaluator.proposal_policy(),
            evaluator.geometry_budget()
        );
    }
    let schedule = weighted_sizes(&args.sizes, &args.size_weights)?;
    let boards: Vec<_> = schedule
        .iter()
        .map(|&size| {
            if !(9..=19).contains(&size) {
                return Err(format!(
                    "Training board sizes must be in 9..=19, got {size}"
                ));
            }
            Board::new(size)
        })
        .collect::<Result<_, _>>()?;
    let preview = args.output.as_os_str() == "-";
    if !preview {
        fs::create_dir_all(&args.output)?;
    }
    let mut stdout = io::stdout().lock();
    let started = Instant::now();
    let mut total_moves = 0;
    let mut completed = 0;
    let mut draws = 0;
    let mut cacheable_games = 0;
    let mut emit = |index: u32,
                    record: GameRecord,
                    generation_seconds: f64|
     -> Result<(), String> {
        total_moves += record.metadata.policies.len();
        completed += usize::from(record.metadata.result.is_some());
        if matches!(record.metadata.result, Some(territory::Outcome::Draw)) {
            draws += 1;
            eprintln!(
                "WARNING DRAW: game={} size={} seed={}",
                index + 1,
                record.metadata.board_size,
                record.metadata.seed
            );
        } else if record.metadata.result.is_none() {
            eprintln!(
                "WARNING TRUNCATED: game={} size={} seed={} max_moves={}",
                index + 1,
                record.metadata.board_size,
                record.metadata.seed,
                args.max_moves
            );
        }
        let output_started = Instant::now();
        if preview {
            writeln!(stdout, "{}", record.sgf).map_err(|e| e.to_string())?;
            stdout.flush().map_err(|e| e.to_string())?;
        } else {
            let path = save_record(&args.output, index + 1, &record).map_err(|e| e.to_string())?;
            if !matches!(args.engine, Engine::Random) && record.metadata.result.is_some() {
                cacheable_games += 1;
            }
            args.nn_view.render(&path, true)?;
        }
        let output_seconds = output_started.elapsed().as_secs_f64();
        eprintln!(
            "game {}/{}: size={}, seed={}, moves={}, finished={}, {:.3}s (generation={:.3}s, output={:.6}s)",
            index + 1,
            args.games,
            record.metadata.board_size,
            record.metadata.seed,
            record.metadata.policies.len(),
            record.metadata.result.is_some(),
            generation_seconds + output_seconds,
            generation_seconds,
            output_seconds,
        );
        if config.search.rollout.is_some() {
            eprintln!(
                "rollouts game {}: {:?}",
                index + 1,
                record.metadata.rollout_stats
            );
        }
        if let Some(profile) = &record.metadata.profile {
            eprintln!("{}", profile.summary());
        }
        Ok(())
    };
    let parallel_stats = if parallel.concurrent_games.min(args.games as usize) > 1 {
        Some(generate_parallel(
            &boards,
            &config,
            evaluator.take().unwrap(),
            parallel,
            &mut emit,
        )?)
    } else {
        for index in 0..args.games {
            let board = boards[index as usize % boards.len()];
            let seed = args.seed.wrapping_add(u64::from(index));
            let game_started = Instant::now();
            let record = if args.profile {
                generate_profiled(board, &config, seed)?
            } else if let Some(evaluator) = &mut evaluator {
                generate_neural(board, &config, seed, evaluator)?
            } else {
                generate(board, &config, seed)?
            };
            emit(index, record, game_started.elapsed().as_secs_f64())?;
        }
        None
    };
    let elapsed = started.elapsed().as_secs_f64();
    eprintln!(
        "{} games ({} completed), {} moves in {:.3}s ({:.2} moves/s)",
        args.games,
        completed,
        total_moves,
        elapsed,
        total_moves as f64 / elapsed
    );
    eprintln!(
        "outcomes: draws={draws} truncated={}",
        args.games as usize - completed
    );
    if let Some(stats) = &parallel_stats {
        eprintln!("native inference: {stats:?}");
    } else if let Some(evaluator) = &evaluator {
        eprintln!("native inference: {:?}", evaluator.stats);
    }
    if let Some(path) = &args.stats_json {
        let stats = parallel_stats
            .as_ref()
            .or_else(|| evaluator.as_ref().map(|e| &e.stats));
        let summary = serde_json::json!({
            "games": args.games, "completed": completed, "draws": draws,
            "truncated": args.games as usize - completed, "moves": total_moves,
            "seconds": elapsed, "moves_per_second": total_moves as f64 / elapsed,
            "inference": stats,
        });
        let file = OpenOptions::new().write(true).create_new(true).open(path)?;
        serde_json::to_writer_pretty(file, &summary)?;
    }
    // Keep search timings/RNG separate. Skip stdout-only, random and wholly
    // truncated jobs; there are no eligible on-disk training games to cache.
    if cacheable_games > 0 {
        args.feature_cache.build(&args.output)?;
    }
    Ok(())
}

fn save_record(
    directory: &Path,
    number: u32,
    record: &GameRecord,
) -> Result<PathBuf, Box<dyn Error>> {
    let stem = format!(
        "game-{number:06}-size{}-seed{}",
        record.metadata.board_size, record.metadata.seed
    );
    let sgf_path = directory.join(format!("{stem}.sgf"));
    let mut sgf_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&sgf_path)?;
    sgf_file.write_all(record.sgf.as_bytes())?;
    Ok(sgf_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn caller_owned_settings_are_required_without_native_defaults() {
        let command = Args::command();
        for name in [
            "simulations",
            "cpuct",
            "widening",
            "widening_exponent",
            "proposal_attempts",
            "independent_exploration",
            "score_weight",
            "komi",
            "komi_stddev",
            "max_moves",
            "policy_simulations",
            "policy_search_probability",
            "chosen_move_temperature",
            "chosen_move_temperature_early",
            "chosen_move_temperature_halflife",
            "sizes",
            "size_weights",
            "concurrent_games",
            "inference_batch_size",
            "inference_batch_wait_ms",
            "inference_threads",
            "feature_threads",
            "cache_positions",
        ] {
            let arg = command
                .get_arguments()
                .find(|arg| arg.get_id() == name)
                .unwrap();
            assert!(arg.is_required_set(), "{name}");
            assert!(arg.get_default_values().is_empty(), "{name}");
        }
        assert!(Args::try_parse_from(["selfplay"]).is_err());
    }
}
