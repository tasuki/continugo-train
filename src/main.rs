// Parallel rules queries create many short-lived vectors. Keep their allocations
// thread-local rather than contending on the system allocator's shared arenas.
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

use clap::Parser;
use continugo_rules::{Board, territory};
use continugo_train::{
    inference::{Evaluator, InferenceDevice, OnnxEvaluator, batching::BatchingConfig},
    sampling::{DEFAULT_SIZE_WEIGHTS, weighted_sizes},
    search::{SearchConfig, rollout::RolloutArgs},
    selfplay::{
        Engine, GameRecord, SelfplayConfig, generate, generate_neural, generate_profiled,
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
    about = "ContinuGo random, unguided MCTS and native ONNX neural MCTS self-play"
)]
struct Args {
    #[arg(long, value_enum, default_value = "mcts")]
    engine: Engine,
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
    games: u32,
    /// Supported board equivalents; allocation follows --size-weights.
    #[arg(long, value_delimiter = ',', default_value = "9,13,19", num_args = 1..)]
    sizes: Vec<i32>,
    /// Board mix: default 40%/40%/20%; 'uniform' preserves literal size cycling.
    #[arg(long, default_value = DEFAULT_SIZE_WEIGHTS)]
    size_weights: String,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 512)]
    max_moves: usize,
    #[arg(long, default_value_t = 32)]
    simulations: u32,
    #[arg(long, default_value_t = 1.5)]
    cpuct: f64,
    #[arg(long, default_value_t = 2.0)]
    widening: f64,
    #[arg(long, default_value_t = 0.5)]
    widening_exponent: f64,
    #[arg(long, default_value_t = 128)]
    proposal_attempts: usize,
    #[arg(long, default_value_t = 0.25)]
    independent_exploration: f64,
    #[command(flatten)]
    rollout: RolloutArgs,
    /// ONNX model directory containing encoder.onnx and policy.onnx (neural engine).
    #[arg(long)]
    model: Option<PathBuf>,
    /// Native ONNX Runtime >=1.20 shared library; otherwise use ORT_DYLIB_PATH.
    #[arg(long)]
    ort_library: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    inference_threads: usize,
    /// Active independent neural games, sharing one model and feature pool (1..=64).
    #[arg(long, default_value_t = 1)]
    concurrent_games: usize,
    /// Maximum inference batch, bucketed by resolution (1..=32).
    #[arg(long, default_value_t = 8)]
    inference_batch_size: usize,
    /// Maximum collection delay; partial batches always run (0..=1000 ms).
    #[arg(long, default_value_t = 2)]
    inference_batch_wait_ms: u64,
    /// Bounded CPU pool for exact coordinate features (1 = serial).
    #[arg(long, default_value_t = 6)]
    feature_threads: usize,
    /// Maximum cached coarse-network contexts; 0 disables, payload capped at 32 MiB.
    #[arg(long, default_value_t = 64)]
    cache_positions: usize,
    #[arg(long, value_enum, default_value = "cuda")]
    device: InferenceDevice,
    /// Time random-engine stages and moves; embed diagnostics in SGF comments.
    #[arg(long)]
    profile: bool,
    /// Write machine-readable run totals and inference stages to a new JSON file.
    #[arg(long)]
    stats_json: Option<PathBuf>,
    /// Directory for self-contained .sgf records, or '-' for stdout.
    #[arg(long, default_value = "-")]
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if args.profile && !matches!(args.engine, Engine::Random) {
        return Err("--profile currently requires --engine random".into());
    }
    let parallel = ParallelConfig {
        games: args.games,
        seed: args.seed,
        concurrent_games: args.concurrent_games,
        batching: BatchingConfig {
            max_batch: args.inference_batch_size,
            max_wait: Duration::from_millis(args.inference_batch_wait_ms),
        },
    };
    parallel.validate()?;
    if !matches!(args.engine, Engine::Neural)
        && (args.concurrent_games != 1
            || args.inference_batch_size != 8
            || args.inference_batch_wait_ms != 2)
    {
        return Err("concurrent games/inference batching require --engine neural".into());
    }
    let config = SelfplayConfig {
        engine: args.engine,
        max_moves: args.max_moves,
        search: SearchConfig {
            simulations: args.simulations,
            cpuct: args.cpuct,
            widening: args.widening,
            widening_exponent: args.widening_exponent,
            proposal_attempts: args.proposal_attempts,
            independent_exploration: args.independent_exploration,
            rollout: args.rollout.config(),
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
        eprintln!(
            "ONNX device={} model={} threads={} feature_threads={} cache_positions={} concurrent_games={} batch_size={} batch_wait_ms={}",
            evaluator.device.name(),
            evaluator.model_id(),
            args.inference_threads,
            args.feature_threads,
            args.cache_positions,
            args.concurrent_games,
            args.inference_batch_size,
            args.inference_batch_wait_ms
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
            save_record(&args.output, index + 1, &record).map_err(|e| e.to_string())?;
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
    let parallel_stats = if args.concurrent_games > 1 {
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
    Ok(())
}

fn save_record(directory: &Path, number: u32, record: &GameRecord) -> Result<(), Box<dyn Error>> {
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
    Ok(())
}
