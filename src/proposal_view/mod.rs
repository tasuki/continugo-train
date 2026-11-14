//! Read-only saved-position diagnostic using the shared ContinuGo embedded viewer.
use crate::{
    geometry_proposals as gp,
    inference::{Evaluator, InferenceDevice, OnnxEvaluator},
    replay::load_position,
};
use continugo_rules::Color;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    io::Write as _,
    path::{Path, PathBuf},
    time::Instant,
};

/// Command-line options for the read-only proposal report.
#[derive(clap::Args)]
pub struct Options {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    ort_library: Option<PathBuf>,
    #[arg(long, value_enum, default_value = "cpu")]
    device: InferenceDevice,
    #[arg(long)]
    sgf: PathBuf,
    #[arg(long)]
    ply: usize,
    /// New directory; never overwrites an existing visualization.
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = gp::DEFAULT_BUDGET)]
    budget: usize,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Built shared viewer JS; defaults to ../continugo/frontend/viewer/dist/continugo-viewer.js.
    #[arg(long)]
    viewer_bundle: Option<PathBuf>,
}

fn save(path: &Path, contents: &[u8]) -> std::io::Result<()> {
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?
        .write_all(contents)
}

fn render(data: &Value) -> Result<String, serde_json::Error> {
    // SGF/comments may contain HTML, including closing script tags. Keep the
    // inline JSON inert; all user-facing labels are assigned with textContent.
    let json = serde_json::to_string(data)?
        .replace('<', "\\u003c")
        .replace('\u{2028}', "\\u2028")
        .replace('\u{2029}', "\\u2029");
    Ok(include_str!("page.html").replacen("__PROPOSAL_DATA_JSON__", &json, 1))
}

/// Generate a portable report directory and return its diagnostic summary.
pub fn generate(args: Options) -> Result<Value, Box<dyn std::error::Error>> {
    if args.output.exists() {
        return Err("--output must be a new directory".into());
    }
    let bundle_path = args.viewer_bundle.clone().unwrap_or_else(|| {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../continugo/frontend/viewer/dist/continugo-viewer.js")
    });
    let bundle = std::fs::read(&bundle_path).map_err(|e| {
        format!(
            "cannot read viewer bundle {}: {e}. Build it in ../continugo with \
            `mise run //frontend:build-viewer`, or supply --viewer-bundle PATH",
            bundle_path.display()
        )
    })?;
    let bundle_hash = format!("{:x}", Sha256::digest(&bundle));
    let text = std::fs::read_to_string(&args.sgf)?;
    let game = load_position(&text, args.ply)?.game;
    let mut evaluator =
        OnnxEvaluator::load(&args.model, args.ort_library.as_deref(), 1, args.device)?;
    evaluator.configure_features(1, 0)?;
    evaluator.configure_geometry(Some(args.budget))?;
    let resolution =
        crate::features::resolution(game.board().diameter(), evaluator.samples_per_diameter())?;
    let started = Instant::now();
    let list = evaluator.geometry_shortlist(&game, args.seed)?;
    let generation_seconds = started.elapsed().as_secs_f64();
    // Authoritative validation only for the final shortlist, no full-domain scan.
    let prepared = game.prepare();
    let legal: Vec<_> = list
        .candidates
        .iter()
        .filter(|c| prepared.play(c.spot()).is_ok())
        .collect();
    let illegal = list.candidates.len() - legal.len();
    let player = if game.on_move() == Color::Black {
        "black"
    } else {
        "white"
    };
    let title = format!("Geometry proposals — ply {} — {} to move", args.ply, player);
    let subtitle = format!(
        "{} × {} board · {} selected cells · soft budget {} / cell · {} illegal proposals omitted",
        game.board().size(),
        game.board().size(),
        list.selected_cells.len(),
        args.budget,
        illegal
    );
    let cells: Vec<_> = list
        .selected_cells
        .iter()
        .map(|&c| gp::cell_rect(c, resolution))
        .collect();
    let stones: Vec<_> = game
        .position()
        .stones()
        .iter()
        .map(|s| {
            json!({
                "x": s.spot.x, "y": s.spot.y,
                "color": if s.color == Color::Black { "black" } else { "white" },
            })
        })
        .collect();
    let page = render(&json!({
        "sgf": text, "move": args.ply, "title": title, "subtitle": subtitle,
        "position": {"board_size": game.board().size(), "radius": game.board().stone_radius(),
            "player": player, "stones": stones},
        "candidates": legal, "cells": cells,
    }))?;
    let report = json!({"sgf": args.sgf, "ply": args.ply, "model": args.model,
        "proposal_policy": gp::VERSION,
        "checkpoint_sha256": evaluator.model_id(),
        "viewer_bundle": bundle_path, "viewer_bundle_sha256": bundle_hash,
        "seed": args.seed, "budget": args.budget, "resolution": resolution,
        "generation_and_encoder_seconds": generation_seconds, "illegal": illegal,
        "legal": legal, "shortlist": list, "stats": evaluator.stats});
    std::fs::create_dir_all(args.output.parent().unwrap_or(Path::new(".")))?;
    std::fs::create_dir(&args.output)?;
    save(&args.output.join("continugo-viewer.js"), &bundle)?;
    save(&args.output.join("index.html"), page.as_bytes())?;
    save(
        &args.output.join("proposals.json"),
        &serde_json::to_vec_pretty(&report)?,
    )?;
    Ok(
        json!({"viewer": args.output.join("index.html"), "legal": legal.len(),
        "illegal": illegal, "preliminary": list.preliminary_candidates,
        "overflow": list.protected_overflow, "generation_and_encoder_seconds": generation_seconds}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_json_cannot_escape_its_script_element() {
        let attack = "</script><script>globalThis.pwned=true</script>\u{2028}\u{2029}";
        let page = render(&json!({"sgf": attack, "title": "<b>untrusted</b>"})).unwrap();
        assert!(!page.contains(attack));
        assert!(!page.contains("__PROPOSAL_DATA_JSON__"));
        let json = page
            .split("<script id=\"proposal-data\" type=\"application/json\">")
            .nth(1)
            .unwrap()
            .split("</script>")
            .next()
            .unwrap();
        let decoded: Value = serde_json::from_str(json).unwrap();
        assert_eq!(decoded["sgf"], attack);
        assert_eq!(decoded["title"], "<b>untrusted</b>");
    }
}
