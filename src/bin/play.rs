//! Persistent JSON-lines move worker for the local browser server.
use clap::Parser;
use continugo_train::{
    inference::{InferenceDevice, OnnxEvaluator},
    replay::load_position,
    search::{SearchConfig, SearchParameters, search_neural},
    selfplay::neural_move_comment,
};
use rand::{SeedableRng, rngs::StdRng};
use serde::Deserialize;
use serde_json::json;
use std::{
    io::{self, BufRead, Write},
    path::PathBuf,
};

#[derive(Parser)]
#[command(args_override_self = true)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    ort_library: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "cpu")]
    device: InferenceDevice,
    #[arg(long)]
    simulations: u32,
    #[command(flatten)]
    search: SearchParameters,
    #[arg(long)]
    inference_threads: usize,
    #[arg(long)]
    feature_threads: usize,
    #[arg(long)]
    cache_positions: usize,
}

#[derive(Deserialize)]
struct Request {
    sgf: String,
    ply: usize,
    simulations: Option<u32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let config = args.search.config(args.simulations);
    config.validate()?;
    let mut evaluator = OnnxEvaluator::load(
        &args.model,
        args.ort_library.as_deref(),
        args.inference_threads,
        args.device,
    )?;
    evaluator.configure_features(args.feature_threads, args.cache_positions)?;
    let mut rng = StdRng::from_os_rng();
    eprintln!("Model ready");
    for line in io::stdin().lock().lines() {
        let line = line?;
        let response = (|| -> Result<_, String> {
            let request: Request = serde_json::from_str(&line).map_err(|e| e.to_string())?;
            let config = match request.simulations {
                Some(simulations @ 1..=1024) => SearchConfig {
                    simulations,
                    ..config.clone()
                },
                Some(_) => return Err("Simulations must be between 1 and 1024".into()),
                None => config.clone(),
            };
            let position = load_position(&request.sgf, request.ply)?;
            if position.game.is_finished() {
                return Err("Game is finished".into());
            }
            let result = search_neural(&position.game, &config, &mut evaluator, &mut rng)?;
            let action = result.best(&mut rng);
            let child = action.apply(&position.game)?;
            let comment = neural_move_comment(&child, &mut evaluator, &result.policy)?;
            Ok(json!({
                "move": action.spot().map(|s| json!({"x": s.x, "y": s.y})),
                "comment": comment,
            }))
        })()
        .unwrap_or_else(|error| json!({"error": error}));
        println!("{response}");
        io::stdout().flush()?;
    }
    Ok(())
}
