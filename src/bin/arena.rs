//! Native paired-color candidate/incumbent matches. SGF remains authoritative.
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

use clap::Parser;
use continugo_rules::{Board, Color, territory};
use continugo_train::{
    inference::{Evaluator, InferenceDevice, OnnxEvaluator, batching::BatchingConfig},
    nn_html::NnHtmlArgs,
    sampling::{DEFAULT_SIZE_WEIGHTS, weighted_sizes},
    search::{SearchConfig, rollout::RolloutArgs},
    selfplay::{
        Engine, REFERENCE_ID, SelfplayConfig, arena::generate_arena, parallel::ParallelConfig,
    },
};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
    time::{Duration, Instant},
};

#[derive(Parser)]
#[command(about = "Fixed-budget paired-color neural matches; no forced scores at move caps")]
struct Args {
    #[arg(
        long,
        required_unless_present = "unguided",
        conflicts_with = "unguided"
    )]
    candidate: Option<PathBuf>,
    /// Diagnostic network-independent MCTS versus the direct reference (not rated).
    #[arg(long, requires = "reference_random", conflicts_with = "incumbent")]
    unguided: bool,
    #[arg(
        long,
        required_unless_present = "reference_random",
        conflicts_with = "reference_random"
    )]
    incumbent: Option<PathBuf>,
    /// Direct random-but-not-eye-filling reference; no MCTS on its turns.
    #[arg(long)]
    reference_random: bool,
    /// Load and validate both models, then exit without creating output or games.
    #[arg(long)]
    check_models: bool,
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
    /// Fixed compensation awarded to White in every game of the pair.
    #[arg(long, default_value_t = 7.0, allow_hyphen_values = true)]
    komi: f64,
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
    /// Score fraction: 1 maximizes expected final score; 0 uses outcome only.
    #[arg(long, default_value_t = 0.5)]
    score_weight: f64,
    #[command(flatten)]
    rollout: RolloutArgs,
    #[arg(long)]
    ort_library: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "cuda")]
    device: InferenceDevice,
    #[arg(long, default_value_t = 1)]
    inference_threads: usize,
    /// Independent games sharing one inference pipeline per model (1..=64).
    #[arg(long, default_value_t = 32)]
    concurrent_games: usize,
    /// Maximum inference batch per model (1..=32), bucketed by resolution.
    #[arg(long, default_value_t = 32)]
    inference_batch_size: usize,
    /// Maximum collection delay; partial batches always run (0..=1000 ms).
    #[arg(long, default_value_t = 2)]
    inference_batch_wait_ms: u64,
    #[arg(long, default_value_t = 6)]
    feature_threads: usize,
    #[arg(long, default_value_t = 64)]
    cache_positions: usize,
    #[command(flatten)]
    nn_view: NnHtmlArgs,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    args.nn_view.validate(true, args.incumbent.is_some())?;
    if args.nn_view.nn_html && args.unguided {
        return Err("--nn-html requires a neural candidate".into());
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
        komi: args.komi,
        komi_stddev: 0.0,
        engine: if args.unguided {
            Engine::Mcts
        } else {
            Engine::Neural
        },
        max_moves: args.max_moves,
        policy_search: None,
        search: SearchConfig {
            simulations: args.simulations,
            cpuct: args.cpuct,
            widening: args.widening,
            widening_exponent: args.widening_exponent,
            proposal_attempts: args.proposal_attempts,
            independent_exploration: args.independent_exploration,
            score_weight: args.score_weight,
            bootstrap_avoid_eyes: false,
            rollout: args.rollout.config(),
            ..Default::default()
        },
    };
    config.validate()?;
    let mut candidate = args
        .candidate
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
    if let Some(candidate) = &mut candidate {
        candidate.configure_features(args.feature_threads, args.cache_positions)?;
    }
    if let Some(incumbent) = &mut incumbent {
        incumbent.configure_features(args.feature_threads, args.cache_positions)?;
    }
    if args.check_models {
        return Ok(());
    }
    fs::create_dir_all(&args.output)?;
    let (mut wins, mut losses, mut draws, mut truncated) = (0, 0, 0, 0);
    let candidate_id = candidate
        .as_ref()
        .map_or("unguided-mcts", |e| e.model_id())
        .to_owned();
    let incumbent_id = incumbent
        .as_ref()
        .map_or(REFERENCE_ID, |e| e.model_id())
        .to_owned();
    let boards = schedule
        .iter()
        .map(|&size| Board::new(size))
        .collect::<Result<Vec<_>, _>>()?;
    let started = Instant::now();
    let stats = generate_arena(
        &boards,
        &config,
        candidate,
        incumbent,
        args.game_offset,
        parallel,
        |ordinal, mut record, seconds| {
            let size = schedule[(ordinal / 2) % schedule.len()];
            let seed = args.seed + (ordinal / 2) as u64;
            let candidate_black = ordinal % 2 == 0;
            let color = if candidate_black { "black" } else { "white" };
            record.annotate_evaluation(&binary_id)?;
            if config.search.rollout.is_some() {
                eprintln!(
                    "rollouts game {}: {:?}",
                    ordinal + 1,
                    record.metadata.rollout_stats
                );
            }
            let path = args.output.join(format!(
                "game-{:06}-size{size}-seed{seed}-candidate-{color}.sgf",
                ordinal + 1
            ));
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&path)
                .map_err(|e| e.to_string())?
                .write_all(record.sgf.as_bytes())
                .map_err(|e| e.to_string())?;
            args.nn_view.render(&path, candidate_black)?;
            eprintln!(
                "arena game {}/{end}: size={size} seed={seed} candidate={color} moves={} completed={} elapsed={:.3}s",
                ordinal + 1,
                record.metadata.policies.len(),
                record.metadata.result.is_some(),
                seconds
            );
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
            Ok(())
        },
    )?;
    for (model, stats) in [candidate_id.as_str(), incumbent_id.as_str()]
        .into_iter()
        .zip(stats)
    {
        eprintln!(
            "arena inference {model}: encoder_positions={} encoder_batches={} encoder_max_batch={} scorer_max_batch={}",
            stats.encoder_positions,
            stats.encoder_batches,
            stats.encoder_max_batch,
            stats.scorer_max_batch
        );
    }
    eprintln!(
        "arena elapsed={:.3}s concurrent_games={}",
        started.elapsed().as_secs_f64(),
        args.concurrent_games.min(args.games as usize)
    );
    println!(
        "arena: candidate={candidate_id} incumbent={incumbent_id} wins={wins} losses={losses} draws={draws} truncated={truncated}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_fixed_komi_defaults_to_seven_and_accepts_zero() {
        let base = [
            "arena",
            "--candidate",
            "model",
            "--reference-random",
            "--output",
            "games",
        ];
        assert_eq!(Args::parse_from(base).komi, 7.0);
        assert_eq!(
            Args::parse_from(base.into_iter().chain(["--komi", "0"])).komi,
            0.0
        );
        assert_eq!(
            Args::parse_from(base.into_iter().chain(["--komi", "-2"])).komi,
            -2.0
        );
    }
}
