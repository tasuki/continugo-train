// Parallel rules queries create many short-lived vectors. Keep their allocations
// thread-local rather than contending on the system allocator's shared arenas.
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

use clap::Parser;
use continugo_rules::Board;
use continugo_train::{
    inference::{Evaluator, InferenceDevice, OnnxEvaluator},
    search::SearchConfig,
    selfplay::{Engine, GameRecord, SelfplayConfig, generate, generate_neural, generate_profiled},
};
use std::{
    error::Error,
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    time::Instant,
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
    /// Equivalent board sizes, cycled in order. The rules currently use integer sizes.
    #[arg(long, value_delimiter = ',', default_value = "9,13,19", num_args = 1..)]
    sizes: Vec<i32>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 512)]
    max_moves: usize,
    #[arg(long, default_value_t = 64)]
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
    /// ONNX model directory containing encoder.onnx and policy.onnx (neural engine).
    #[arg(long)]
    model: Option<PathBuf>,
    /// Native ONNX Runtime >=1.20 shared library; otherwise use ORT_DYLIB_PATH.
    #[arg(long)]
    ort_library: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    inference_threads: usize,
    /// Bounded CPU pool for exact coordinate features (1 = serial).
    #[arg(long, default_value_t = 4)]
    feature_threads: usize,
    /// Maximum cached coarse-network contexts; 0 disables, payload capped at 32 MiB.
    #[arg(long, default_value_t = 64)]
    cache_positions: usize,
    #[arg(long, value_enum, default_value = "cuda")]
    device: InferenceDevice,
    /// Time random-engine stages and moves; embed diagnostics in SGF comments.
    #[arg(long)]
    profile: bool,
    /// Directory for self-contained .sgf records, or '-' for stdout.
    #[arg(long, default_value = "-")]
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if args.profile && !matches!(args.engine, Engine::Random) {
        return Err("--profile currently requires --engine random".into());
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
            "ONNX device={} model={} threads={} feature_threads={} cache_positions={}",
            evaluator.device.name(),
            evaluator.model_id(),
            args.inference_threads,
            args.feature_threads,
            args.cache_positions
        );
    }
    let boards: Vec<_> = args
        .sizes
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
        let generation_seconds = game_started.elapsed().as_secs_f64();
        total_moves += record.metadata.policies.len();
        completed += usize::from(record.metadata.result.is_some());
        let output_started = Instant::now();
        if preview {
            writeln!(stdout, "{}", record.sgf)?;
            stdout.flush()?;
        } else {
            save_record(&args.output, index + 1, &record)?;
        }
        let output_seconds = output_started.elapsed().as_secs_f64();
        eprintln!(
            "game {}/{}: size={}, seed={}, moves={}, finished={}, {:.3}s (generation={:.3}s, output={:.6}s)",
            index + 1,
            args.games,
            board.size(),
            seed,
            record.metadata.policies.len(),
            record.metadata.result.is_some(),
            game_started.elapsed().as_secs_f64(),
            generation_seconds,
            output_seconds,
        );
        if let Some(profile) = &record.metadata.profile {
            eprintln!("{}", profile.summary());
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    eprintln!(
        "{} games ({} completed), {} moves in {:.3}s ({:.2} moves/s)",
        args.games,
        completed,
        total_moves,
        elapsed,
        total_moves as f64 / elapsed
    );
    if let Some(evaluator) = &evaluator {
        eprintln!("native inference: {:?}", evaluator.stats);
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
