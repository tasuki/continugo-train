//! Emit one ephemeral CGFEAT01 tensor sample, not a replacement game record.
//! See FEATURES.md for schema, binary layout and label semantics.

use clap::Parser;
use continugo_rules::{Action, COORDINATE_ACTIONS};
use continugo_train::{
    features::{COARSE_PLANES, FeatureExtractor, MOVE_FEATURES, SCHEMA_VERSION},
    replay,
};
use std::{
    collections::HashSet,
    error::Error,
    fs::{self, OpenOptions},
    io::{self, BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

#[derive(Parser)]
#[command(about = "Extract real neural features from an SGF pre-move position")]
struct Args {
    #[arg(long)]
    sgf: PathBuf,
    /// Number of moves already played (zero-based pre-move position).
    #[arg(long, default_value_t = 0)]
    ply: usize,
    #[arg(long, default_value_t = 4)]
    samples_per_diameter: usize,
    /// Coordinate subset for diagnostics. Omit for the full million-action domain.
    #[arg(long, value_delimiter = ',', num_args = 1.., value_parser = clap::value_parser!(u32).range(0..COORDINATE_ACTIONS as i64))]
    actions: Option<Vec<u32>>,
    /// Binary tensor output, or '-' for stdout. Existing files are not overwritten.
    #[arg(long, default_value = "-")]
    output: PathBuf,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let started = Instant::now();
    let sample = replay::load_position(&fs::read_to_string(&args.sgf)?, args.ply)?;
    let replay_seconds = started.elapsed().as_secs_f64();
    let full_domain = args.actions.is_none();
    let actions: Vec<_> = args
        .actions
        .unwrap_or_else(|| (0..COORDINATE_ACTIONS).collect())
        .into_iter()
        .map(|id| Action::from_index(id).expect("clap validated action"))
        .collect();
    if !full_domain {
        let mut unique = HashSet::new();
        if actions.iter().any(|action| !unique.insert(action.index())) {
            return Err("duplicate coordinate action".into());
        }
    }
    let started = Instant::now();
    let extractor = FeatureExtractor::new(&sample.game, args.samples_per_diameter)?;
    let preparation_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let moves = extractor.moves(&actions)?;
    let moves_seconds = started.elapsed().as_secs_f64();
    let started = Instant::now();
    let output: Box<dyn Write> = if args.output.as_os_str() == "-" {
        Box::new(io::stdout().lock())
    } else {
        Box::new(
            OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&args.output)?,
        )
    };
    let mut output = BufWriter::new(output);
    output.write_all(b"CGFEAT01")?;
    let policy = sample.policy.as_deref().unwrap_or_default();
    for number in [
        SCHEMA_VERSION,
        COARSE_PLANES.len() as u32,
        MOVE_FEATURES.len() as u32,
        sample.game.board().size() as u32,
        sample.game.board().diameter() as u32,
        extractor.resolution as u32,
        args.ply as u32,
        replay::player_index(sample.game.on_move()),
        actions.len() as u32,
        policy.len() as u32,
        u32::from(sample.outcome.is_some()),
    ] {
        output.write_all(&number.to_le_bytes())?;
    }
    output.write_all(&sample.outcome.unwrap_or(0.0).to_le_bytes())?;
    output.write_all(&(args.samples_per_diameter as u32).to_le_bytes())?;
    output.write_all(&u32::from(full_domain).to_le_bytes())?;
    for &number in extractor.coarse.iter().chain(&moves.exact) {
        output.write_all(&number.to_le_bytes())?;
    }
    output.write_all(&moves.legal)?;
    for action in &actions {
        output.write_all(&action.index().to_le_bytes())?;
    }
    for target in policy {
        output.write_all(&target.action.to_le_bytes())?;
    }
    for target in policy {
        output.write_all(&target.visits.to_le_bytes())?;
    }
    output.flush()?;
    eprintln!(
        "features: size={}, ply={}, grid={}, coordinates={}, legal={}, replay={replay_seconds:.6}s, coarse/preparation={preparation_seconds:.6}s, exact={moves_seconds:.6}s, output={:.6}s",
        sample.game.board().size(),
        args.ply,
        extractor.resolution,
        actions.len(),
        moves.legal.iter().filter(|&&x| x != 0).count(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
