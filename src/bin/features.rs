//! SGF metadata inspection and CGFEAT01 tensors. Server mode keeps one validated
//! game in memory and frames each tensor/error response without JSON game records.

use clap::Parser;
use continugo_rules::{Action, COORDINATE_ACTIONS};
use continugo_train::{feature_io::encode_position, replay::ReplayGame};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    error::Error,
    fs::{self, OpenOptions},
    io::{self, BufRead, BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Parser)]
#[command(about = "Extract real neural features or inspect validated SGF training metadata")]
struct Args {
    #[arg(long, required_unless_present = "serve")]
    sgf: Option<PathBuf>,
    /// Number of moves already played (zero-based pre-move position).
    #[arg(long, default_value_t = 0)]
    ply: usize,
    #[arg(long, default_value_t = 4)]
    samples_per_diameter: usize,
    #[arg(long, value_delimiter = ',', num_args = 1.., value_parser = clap::value_parser!(u32).range(0..COORDINATE_ACTIONS as i64))]
    actions: Option<Vec<u32>>,
    #[arg(long, default_value = "-")]
    output: PathBuf,
    /// Emit metadata only (not game moves), including eligible training plies.
    #[arg(long, conflicts_with_all = ["serve", "actions", "output", "ply"])]
    inspect: bool,
    /// Read JSON requests on stdin, return status/u64-length framed binary packets.
    #[arg(long, conflicts_with_all = ["sgf", "inspect", "actions", "output", "ply"])]
    serve: bool,
}

fn read_game(path: &Path, expected: Option<&str>) -> Result<(String, ReplayGame), String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let hash = format!("{:x}", Sha256::digest(&bytes));
    if expected.is_some_and(|expected| expected != hash) {
        return Err("SGF changed since dataset inspection".into());
    }
    let game = ReplayGame::load(std::str::from_utf8(&bytes).map_err(|e| e.to_string())?)?;
    Ok((hash, game))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    sgf: PathBuf,
    ply: usize,
    samples_per_diameter: usize,
    file_sha256: Option<String>,
}

fn serve() -> Result<(), Box<dyn Error>> {
    let actions: Vec<_> = (0..COORDINATE_ACTIONS)
        .map(|i| Action::from_index(i).unwrap())
        .collect();
    let mut cached: Option<(PathBuf, String, ReplayGame)> = None;
    let mut output = BufWriter::new(io::stdout().lock());
    for line in io::stdin().lock().lines() {
        let line = line?;
        let response = (|| -> Result<Vec<u8>, String> {
            let request: Request = serde_json::from_str(&line).map_err(|e| e.to_string())?;
            let reuse = cached.as_ref().is_some_and(|(path, hash, _)| {
                *path == request.sgf
                    && request
                        .file_sha256
                        .as_ref()
                        .is_none_or(|expected| expected == hash)
            });
            if !reuse {
                let (hash, game) = read_game(&request.sgf, request.file_sha256.as_deref())?;
                cached = Some((request.sgf, hash, game));
            }
            let sample = cached.as_ref().unwrap().2.position(request.ply)?;
            encode_position(
                &sample,
                request.ply,
                request.samples_per_diameter,
                &actions,
                true,
            )
        })();
        let (status, bytes) = match response {
            Ok(bytes) => (0u8, bytes),
            Err(error) => (1, error.into_bytes()),
        };
        output.write_all(&[status])?;
        output.write_all(&(bytes.len() as u64).to_le_bytes())?;
        output.write_all(&bytes)?;
        output.flush()?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    if args.serve {
        return serve();
    }
    let started = Instant::now();
    let (hash, game) = read_game(args.sgf.as_ref().unwrap(), None)?;
    if args.inspect {
        #[derive(Serialize)]
        struct Info {
            #[serde(flatten)]
            game: continugo_train::replay::GameInfo,
            file_sha256: String,
        }
        println!(
            "{}",
            serde_json::to_string(&Info {
                game: game.info(args.samples_per_diameter)?,
                file_sha256: hash
            })?
        );
        return Ok(());
    }
    let sample = game.position(args.ply)?;
    let replay_seconds = started.elapsed().as_secs_f64();
    let full_domain = args.actions.is_none();
    let actions: Vec<_> = args
        .actions
        .unwrap_or_else(|| (0..COORDINATE_ACTIONS).collect())
        .into_iter()
        .map(|i| Action::from_index(i).unwrap())
        .collect();
    if !full_domain {
        let mut seen = HashSet::new();
        if actions.iter().any(|a| !seen.insert(a.index())) {
            return Err("duplicate coordinate action".into());
        }
    }
    let started = Instant::now();
    let packet = encode_position(
        &sample,
        args.ply,
        args.samples_per_diameter,
        &actions,
        full_domain,
    )?;
    let extraction = started.elapsed().as_secs_f64();
    let started = Instant::now();
    if args.output.as_os_str() == "-" {
        io::stdout().lock().write_all(&packet)?;
    } else {
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&args.output)?
            .write_all(&packet)?;
    }
    eprintln!(
        "features: size={}, ply={}, coordinates={}, replay={replay_seconds:.6}s, extraction+encode={extraction:.6}s, output={:.6}s",
        sample.game.board().size(),
        args.ply,
        actions.len(),
        started.elapsed().as_secs_f64()
    );
    Ok(())
}
