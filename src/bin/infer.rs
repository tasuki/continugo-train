//! One-position diagnostic: selective hierarchical log probabilities and value.
//! Unselected and illegal coordinates print as null; this is not a game record.
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

use clap::Parser;
use continugo_train::{
    action::Action,
    inference::{Evaluator, InferenceDevice, OnnxEvaluator},
    replay::load_position,
};
use std::{path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    ort_library: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    threads: usize,
    #[arg(long, default_value_t = 6)]
    feature_threads: usize,
    #[arg(long, default_value_t = 64)]
    cache_positions: usize,
    #[arg(long, value_enum, default_value = "cuda")]
    device: InferenceDevice,
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
    #[arg(long, default_value_t = 0)]
    warmup: usize,
    #[arg(long, default_value_t = 1)]
    iterations: usize,
    /// Include measured wall time and structured stage counters in stdout JSON.
    #[arg(long)]
    profile_json: bool,
}

fn report(mut value: serde_json::Value, args: &Args, evaluator: &OnnxEvaluator, seconds: f64) {
    if args.profile_json {
        value["seconds"] = seconds.into();
        value["iterations"] = args.iterations.into();
        value["stats"] =
            serde_json::to_value(&evaluator.stats).expect("finite inference statistics");
    }
    println!("{value}");
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
    if args.iterations == 0 {
        return Err("iterations must be positive".into());
    }
    let mut evaluator = OnnxEvaluator::load(
        &args.model,
        args.ort_library.as_deref(),
        args.threads,
        args.device,
    )?;
    evaluator.configure_features(args.feature_threads, args.cache_positions)?;
    let text = std::fs::read_to_string(&args.sgf)?;
    let position = load_position(&text, args.ply)?;
    for _ in 0..args.warmup {
        if args.value_only {
            evaluator.value(&position.game)?;
        } else {
            evaluator.evaluate(&position.game)?;
        }
    }
    evaluator.stats = Default::default();
    let started = Instant::now();
    if args.value_only {
        let mut value = 0.0;
        for _ in 0..args.iterations {
            value = evaluator.value(&position.game)?;
        }
        report(
            serde_json::json!({"value": value, "device": args.device.name()}),
            &args,
            &evaluator,
            started.elapsed().as_secs_f64(),
        );
        eprintln!("{:?}", evaluator.stats);
        return Ok(());
    }
    let mut last = None;
    for _ in 0..args.iterations {
        last = Some(evaluator.evaluate(&position.game)?);
    }
    let seconds = started.elapsed().as_secs_f64();
    let evaluation = last.unwrap();
    let logits: Vec<Option<f32>> = args
        .actions
        .iter()
        .map(|&id| {
            let value = evaluation.logits[id as usize];
            value.is_finite().then_some(value)
        })
        .collect();
    // Diagnostic summaries of selected nonempty cells, not the full legality set.
    let size = continugo_train::features::resolution(
        position.game.board().diameter(),
        evaluator.samples_per_diameter(),
    )?;
    let cells: std::collections::BTreeSet<_> = evaluation
        .legal
        .iter()
        .enumerate()
        .filter(|(_, legal)| **legal != 0)
        .map(|(id, _)| continugo_train::cells::cell(id as u32, size))
        .collect();
    let policy_mass: f64 = evaluation.logits.iter().map(|&p| f64::from(p).exp()).sum();
    report(
        serde_json::json!({"value": evaluation.value, "actions": args.actions, "logits": logits, "device": args.device.name(), "cells": cells, "policy_mass": policy_mass}),
        &args,
        &evaluator,
        seconds,
    );
    eprintln!("{:?}", evaluator.stats);
    Ok(())
}
