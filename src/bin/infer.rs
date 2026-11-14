//! One-position diagnostic: selective hierarchical log probabilities and value.
//! Unselected and illegal coordinates print as null; this is not a game record.
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

use clap::Parser;
use continugo_train::{
    action::Action,
    inference::{Evaluator, InferenceDevice, OnnxEvaluator},
    replay::load_position,
    search::{SearchConfig, search_neural},
};
use rand::{SeedableRng, rngs::StdRng};
use sha2::{Digest, Sha256};
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
    /// Override the schema-5 geometry soft budget (default 16, range 1..=128).
    #[arg(long, conflicts_with = "complete_cells")]
    geometry_budget: Option<usize>,
    /// Analyze complete selected cells instead of using geometry proposals.
    #[arg(long)]
    complete_cells: bool,
    #[arg(long)]
    value_only: bool,
    #[arg(long, default_value_t = 0)]
    warmup: usize,
    #[arg(long, default_value_t = 1)]
    iterations: usize,
    /// Include measured wall time and structured stage counters in stdout JSON.
    #[arg(long)]
    profile_json: bool,
    /// Run a seeded root search and label its visits after search (diagnostic only).
    #[arg(long, conflicts_with = "value_only")]
    simulations: Option<u32>,
    #[arg(long, default_value_t = 0.5)]
    score_weight: f64,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn report(mut value: serde_json::Value, args: &Args, evaluator: &OnnxEvaluator, seconds: f64) {
    value["proposal_policy"] = evaluator.proposal_policy().into();
    if let Some(budget) = evaluator.geometry_budget() {
        value["geometry_budget"] = budget.into();
    }
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
    if args.complete_cells {
        evaluator.configure_geometry(None)?;
    } else if let Some(budget) = args.geometry_budget {
        evaluator.configure_geometry(Some(budget))?;
    }
    evaluator.profile_coarse = args.profile_json;
    let text = std::fs::read_to_string(&args.sgf)?;
    let position = load_position(&text, args.ply)?;
    if let Some(simulations) = args.simulations {
        let config = SearchConfig {
            simulations,
            score_weight: args.score_weight,
            ..Default::default()
        };
        let started = Instant::now();
        let estimate = evaluator.estimate(&position.game)?;
        let mut rng = StdRng::seed_from_u64(args.seed);
        let result = search_neural(&position.game, &config, &mut evaluator, &mut rng)?;
        let search_seconds = started.elapsed().as_secs_f64();
        let best = result.best(&mut rng);
        let mut eyes = continugo_analysis::SmallEyeAnalysis::new(position.game.position());
        let mut eye_visits = 0;
        let mut exposed_visits = 0;
        let mut pass_visits = 0;
        let mut visits = Vec::new();
        for visit in result.policy {
            let mut eye_fill = false;
            let mut capture_exposed = false;
            if let Some(spot) = visit.action.spot() {
                let child = position.game.play(spot)?;
                let captures =
                    child.position().stones().len() < position.game.position().stones().len() + 1;
                eye_fill = !captures && eyes.fills_small_eye(position.game.on_move(), spot);
                let group = child
                    .position()
                    .groups(position.game.on_move())
                    .into_iter()
                    .find(|members| {
                        members
                            .iter()
                            .any(|&i| child.position().stones()[i].spot == spot)
                    })
                    .ok_or("diagnostic placement group missing")?;
                capture_exposed = child
                    .position()
                    .prepare()
                    .analyze_group(&group)
                    .capture_in_one
                    .is_some();
            } else {
                pass_visits += visit.visits;
            }
            if eye_fill {
                eye_visits += visit.visits;
            }
            if capture_exposed {
                exposed_visits += visit.visits;
            }
            visits.push(
                serde_json::json!({"action": visit.action.index(), "visits": visit.visits,
                "eye_fill": eye_fill, "capture_exposed": capture_exposed}),
            );
        }
        let output = serde_json::json!({"value": estimate.value, "score": estimate.score,
            "utility": estimate.utility(config.score_weight)?, "search": config, "seed": args.seed,
            "search_seconds": search_seconds, "best": best.index(), "visits": visits,
            "sgf_sha256": format!("{:x}", Sha256::digest(text.as_bytes())), "ply": args.ply,
            "pass_visit_fraction": pass_visits as f64 / simulations as f64,
            "eye_fill_visit_fraction": eye_visits as f64 / simulations as f64,
            "capture_exposed_visit_fraction": exposed_visits as f64 / simulations as f64,
            "checkpoint_sha256": evaluator.model_id()});
        report(output, &args, &evaluator, started.elapsed().as_secs_f64());
        return Ok(());
    }
    for _ in 0..args.warmup {
        if args.value_only {
            evaluator.value(&position.game)?;
        } else {
            evaluator.evaluate_seeded(&position.game, args.seed)?;
        }
    }
    evaluator.stats = Default::default();
    let started = Instant::now();
    if args.value_only {
        let mut estimate = None;
        for _ in 0..args.iterations {
            estimate = Some(evaluator.estimate(&position.game)?);
        }
        let estimate = estimate.unwrap();
        report(
            serde_json::json!({"value": estimate.value, "score": estimate.score, "device": args.device.name()}),
            &args,
            &evaluator,
            started.elapsed().as_secs_f64(),
        );
        eprintln!("{:?}", evaluator.stats);
        return Ok(());
    }
    let mut last = None;
    for _ in 0..args.iterations {
        last = Some(evaluator.evaluate_seeded(&position.game, args.seed)?);
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
        serde_json::json!({"value": evaluation.value, "score": evaluation.score, "actions": args.actions, "logits": logits, "device": args.device.name(), "cells": cells, "policy_mass": policy_mass}),
        &args,
        &evaluator,
        seconds,
    );
    eprintln!("{:?}", evaluator.stats);
    Ok(())
}
