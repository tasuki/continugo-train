//! Persistent JSON-lines move worker for the local browser server.
use clap::Parser;
use continugo_train::{
    inference::{InferenceDevice, OnnxEvaluator},
    replay::load_position,
    search::{SearchConfig, search_neural},
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
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    ort_library: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "cpu")]
    device: InferenceDevice,
    #[arg(long, default_value_t = 128)]
    simulations: u32,
}

#[derive(Deserialize)]
struct Request {
    sgf: String,
    ply: usize,
    simulations: Option<u32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let config = SearchConfig {
        simulations: args.simulations,
        ..Default::default()
    };
    config.validate()?;
    let mut evaluator =
        OnnxEvaluator::load(&args.model, args.ort_library.as_deref(), 1, args.device)?;
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
