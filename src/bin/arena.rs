//! Native paired-color candidate/incumbent matches. SGF remains authoritative.
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

use clap::Parser;
use continugo_rules::{Board, Color, territory};
use continugo_train::{
    inference::{Evaluator, InferenceDevice, OnnxEvaluator},
    sampling::{DEFAULT_SIZE_WEIGHTS, weighted_sizes},
    search::SearchConfig,
    selfplay::{Engine, REFERENCE_ID, SelfplayConfig, generate_match, generate_reference_match},
};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
};

#[derive(Parser)]
#[command(about = "Fixed-budget paired-color neural matches; no forced scores at move caps")]
struct Args {
    #[arg(long)]
    candidate: PathBuf,
    #[arg(
        long,
        required_unless_present = "reference_random",
        conflicts_with = "reference_random"
    )]
    incumbent: Option<PathBuf>,
    /// Direct random-but-not-eye-filling reference; no MCTS on its turns.
    #[arg(long)]
    reference_random: bool,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 20, value_parser = clap::value_parser!(u32).range(1..))]
    games: u32,
    /// Global game ordinal for restartable partial pairs; even ordinals use candidate black.
    #[arg(long, default_value_t = 0)]
    game_offset: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, value_delimiter = ',', default_value = "9,13,19")]
    sizes: Vec<i32>,
    #[arg(long, default_value = DEFAULT_SIZE_WEIGHTS)]
    size_weights: String,
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
    #[arg(long)]
    ort_library: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "cuda")]
    device: InferenceDevice,
    #[arg(long, default_value_t = 1)]
    inference_threads: usize,
    #[arg(long, default_value_t = 4)]
    feature_threads: usize,
    #[arg(long, default_value_t = 64)]
    cache_positions: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let binary_id = format!("{:x}", Sha256::digest(fs::read(std::env::current_exe()?)?));
    let schedule = weighted_sizes(&args.sizes, &args.size_weights)?;
    let end = args
        .game_offset
        .checked_add(args.games as usize)
        .ok_or("game ordinal overflow")?;
    args.seed
        .checked_add(((end - 1) / 2) as u64)
        .ok_or("pair seed overflow")?;
    let config = SelfplayConfig {
        engine: Engine::Neural,
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
    let mut candidate = OnnxEvaluator::load(
        &args.candidate,
        args.ort_library.as_deref(),
        args.inference_threads,
        args.device,
    )?;
    let mut incumbent = args
        .incumbent
        .as_ref()
        .map(|path| {
            OnnxEvaluator::load(
                path,
                args.ort_library.as_deref(),
                args.inference_threads,
                args.device,
            )
        })
        .transpose()?;
    candidate.configure_features(args.feature_threads, args.cache_positions)?;
    if let Some(incumbent) = &mut incumbent {
        incumbent.configure_features(args.feature_threads, args.cache_positions)?;
    }
    fs::create_dir_all(&args.output)?;
    let (mut wins, mut losses, mut draws, mut truncated) = (0, 0, 0, 0);
    for ordinal in args.game_offset..end {
        let size = schedule[(ordinal / 2) % schedule.len()];
        let seed = args.seed + (ordinal / 2) as u64;
        let candidate_black = ordinal % 2 == 0;
        let color = if candidate_black { "black" } else { "white" };
        let board = Board::new(size)?;
        let mut record = if let Some(incumbent) = &mut incumbent {
            if candidate_black {
                generate_match(board, &config, seed, &mut candidate, incumbent)?
            } else {
                generate_match(board, &config, seed, incumbent, &mut candidate)?
            }
        } else {
            generate_reference_match(board, &config, seed, &mut candidate, candidate_black)?
        };
        record.annotate_evaluation(&binary_id)?;
        let path = args.output.join(format!(
            "game-{:06}-size{size}-seed{seed}-candidate-{color}.sgf",
            ordinal + 1
        ));
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)?
            .write_all(record.sgf.as_bytes())?;
        match record.metadata.result {
            Some(territory::Outcome::Draw) => {
                draws += 1;
                eprintln!(
                    "WARNING DRAW: arena game={} size={size} seed={seed}",
                    ordinal + 1
                );
            }
            Some(territory::Outcome::Win(winner, _)) => {
                if (winner == Color::Black) == candidate_black {
                    wins += 1;
                } else {
                    losses += 1;
                }
            }
            None => {
                truncated += 1;
                eprintln!(
                    "WARNING TRUNCATED: arena game={} size={size} seed={seed} cap={}",
                    ordinal + 1,
                    args.max_moves
                );
            }
        }
    }
    println!(
        "arena: candidate={} incumbent={} wins={wins} losses={losses} draws={draws} truncated={truncated}",
        candidate.model_id(),
        incumbent.as_ref().map_or(REFERENCE_ID, |e| e.model_id())
    );
    Ok(())
}
