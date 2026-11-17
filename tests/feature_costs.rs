//! Opt-in release profiling of coarse plane families. No feature/rule changes.
use continugo_train::{features::FeatureExtractor, replay::load_position};
use serde::Deserialize;
use std::{hint::black_box, time::Instant};

#[derive(Deserialize)]
struct Case {
    sgf: String,
    plies: Vec<usize>,
}

#[test]
#[ignore = "set CONTINUGO_FEATURE_COST_CASES and CONTINUGO_FEATURE_COST_OUTPUT; run --release --nocapture"]
fn coarse_plane_cost_breakdown() {
    let cases: Vec<Case> = serde_json::from_str(
        &std::fs::read_to_string(std::env::var("CONTINUGO_FEATURE_COST_CASES").unwrap()).unwrap(),
    )
    .unwrap();
    let mut rows = Vec::new();
    for case in cases {
        let source = std::fs::read_to_string(&case.sgf).unwrap();
        for ply in case.plies {
            // Replay and scoring are outside the measured region.
            let sample = load_position(&source, ply).unwrap();
            for _ in 0..3 {
                black_box(FeatureExtractor::new(&sample.game, 4).unwrap());
                black_box(FeatureExtractor::profiled(&sample.game, 4).unwrap());
            }
            let mut measurements = Vec::new();
            for repeat in 0..15 {
                let mut plain_seconds = 0.0;
                let mut profiled_seconds = 0.0;
                let mut stages = None;
                let mut plain = None;
                let mut profiled = None;
                for step in 0..2 {
                    let tick = Instant::now();
                    if (repeat + step) % 2 == 0 {
                        let input = FeatureExtractor::new(&sample.game, 4).unwrap();
                        plain_seconds = tick.elapsed().as_secs_f64();
                        plain = Some(input);
                    } else {
                        let (input, profile) = FeatureExtractor::profiled(&sample.game, 4).unwrap();
                        profiled_seconds = tick.elapsed().as_secs_f64();
                        profiled = Some(input);
                        stages = Some(profile);
                    }
                }
                assert_eq!(
                    plain.as_ref().unwrap().coarse,
                    profiled.as_ref().unwrap().coarse
                );
                black_box((&plain, &profiled));
                measurements.push(serde_json::json!({
                    "plain_seconds": plain_seconds,
                    "profiled_seconds": profiled_seconds,
                    "stages": stages.unwrap(),
                }));
            }
            use sha2::{Digest, Sha256};
            let hash = format!("{:x}", Sha256::digest(source.as_bytes()));
            rows.push(serde_json::json!({
                "sgf": case.sgf,
                "sgf_sha256": hash,
                "ply": ply,
                "stones": sample.game.position().stones().len(),
                "resolution": FeatureExtractor::new(&sample.game, 4).unwrap().resolution,
                "measurements": measurements,
            }));
        }
    }
    let output = std::env::var("CONTINUGO_FEATURE_COST_OUTPUT").unwrap();
    std::fs::write(&output, serde_json::to_string_pretty(&rows).unwrap() + "\n").unwrap();
    println!("coarse plane profile: {output}");
}
