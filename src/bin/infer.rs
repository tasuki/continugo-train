//! One-position native diagnostic: selected raw logits/value on stdout as JSON.
//! This is not a game-record writer. Inference still evaluates the full domain.
use clap::Parser;
use continugo_train::{
    action::Action,
    inference::{Evaluator, OnnxEvaluator},
    replay::load_position,
};
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    ort_library: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    threads: usize,
    #[arg(long)]
    sgf: PathBuf,
    #[arg(long, default_value_t = 0)]
    ply: usize,
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "0,999,1000,999999,500500,500501,1000000"
    )]
    actions: Vec<u32>,
    #[arg(long)]
    value_only: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    if args
        .actions
        .iter()
        .any(|&id| Action::from_index(id).is_none())
    {
        return Err("invalid diagnostic action ID".into());
    }
    let mut evaluator =
        OnnxEvaluator::load(&args.model, args.ort_library.as_deref(), args.threads)?;
    let text = std::fs::read_to_string(&args.sgf)?;
    let position = load_position(&text, args.ply)?;
    if args.value_only {
        println!(
            "{}",
            serde_json::json!({"value": evaluator.value(&position.game)?})
        );
        eprintln!("{:?}", evaluator.stats);
        return Ok(());
    }
    let evaluation = evaluator.evaluate(&position.game)?;
    let logits: Vec<Option<f32>> = args
        .actions
        .iter()
        .map(|&id| {
            let value = evaluation.logits[id as usize];
            value.is_finite().then_some(value)
        })
        .collect();
    println!(
        "{}",
        serde_json::json!({"value": evaluation.value, "actions": args.actions, "logits": logits})
    );
    eprintln!("{:?}", evaluator.stats);
    Ok(())
}
