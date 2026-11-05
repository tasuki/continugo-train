//! Opt-in release benchmark; all artifacts stay in continugo-train.
//! See docs/current/profiling.md for invocation and workload definitions.

use super::*;
use crate::replay::ReplayGame;
use continugo_analysis::unconditionally_alive;
use continugo_rules::{Color, Spot};
use rand::{SeedableRng, rngs::StdRng};
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{env, fs, hint::black_box, path::Path, process::Command, time::Duration};

#[derive(Serialize)]
struct Timing {
    median_ms: f64,
    min_ms: f64,
    max_ms: f64,
    samples_ms: Vec<f64>,
    iterations: Vec<usize>,
}

impl Timing {
    fn new(samples: Vec<(f64, usize)>) -> Self {
        let samples_ms: Vec<_> = samples.iter().map(|s| s.0).collect();
        let mut sorted = samples_ms.clone();
        sorted.sort_by(f64::total_cmp);
        let middle = sorted.len() / 2;
        let median_ms = (sorted[middle] + sorted[(sorted.len() - 1) / 2]) / 2.0;
        Self {
            median_ms,
            min_ms: sorted[0],
            max_ms: *sorted.last().unwrap(),
            samples_ms,
            iterations: samples.iter().map(|s| s.1).collect(),
        }
    }
}

// Calibration batches are discarded. Each operation creates fresh analysis;
// warmup warms code/allocator, not a retained position-analysis cache.
fn sample<T>(minimum: Duration, mut operation: impl FnMut() -> T) -> (f64, usize) {
    black_box(operation());
    let mut iterations = 1;
    loop {
        let started = Instant::now();
        for _ in 0..iterations {
            black_box(operation());
        }
        let elapsed = started.elapsed();
        if elapsed >= minimum {
            return (
                elapsed.as_secs_f64() * 1000.0 / iterations as f64,
                iterations,
            );
        }
        iterations = iterations
            .checked_mul(2)
            .expect("benchmark iteration overflow");
    }
}

fn eyes(game: &Game, available: &GeometricAvailability, queries: &[(Color, Spot)]) -> usize {
    let mut analysis = SmallEyeAnalysis::with_availability(game.position(), available);
    queries
        .iter()
        .filter(|&&(color, spot)| analysis.fills_small_eye(color, spot))
        .count()
}

// First run the actual selector, then recover its eye-only workload by replaying
// precisely that proposal prefix. Rule validation and capture exemption stay out
// of the eye-only timer. Assertions guard the trace against selector drift.
fn eye_trace(
    game: &Game,
    available: &GeometricAvailability,
    seed: u64,
) -> (Vec<(Color, Spot)>, MoveProfile) {
    let mut profile = MoveProfile::default();
    profile.passed = random_move::<true>(
        game,
        available,
        &mut StdRng::seed_from_u64(seed),
        &mut profile,
    )
    .is_none();
    let mut rng = StdRng::seed_from_u64(seed);
    let order = Permutation::new(available.coordinates().len() as u32, &mut rng);
    let prepared = game.prepare();
    let enemy = game.on_move().other();
    let enemy_count = game
        .position()
        .stones()
        .iter()
        .filter(|s| s.color == enemy)
        .count();
    let queries: Vec<_> = order
        .take(profile.proposals as usize)
        .filter_map(|i| {
            let spot = available.coordinates()[i as usize].spot().unwrap();
            let next = prepared.play(spot).ok()?;
            // Legal own-eye fills can remove other friendly groups; count the enemy
            // explicitly, as the production random selector does.
            let captured = next
                .position()
                .stones()
                .iter()
                .filter(|s| s.color == enemy)
                .count()
                < enemy_count;
            (!captured).then_some((game.on_move(), spot))
        })
        .collect();
    assert_eq!(queries.len() as u64, profile.eyes.calls);
    assert_eq!(
        eyes(game, available, &queries) as u64,
        profile.rejected_eyes
    );
    (queries, profile)
}

fn positive(name: &str, default: usize) -> usize {
    let value = env::var(name)
        .map(|s| s.parse::<usize>().expect("expected an integer"))
        .unwrap_or(default);
    assert!(value > 0, "{name} must be positive");
    value
}

fn git(path: &Path, arguments: &[&str]) -> String {
    let output = Command::new("git")
        .arg("-C")
        .arg(path)
        .args(arguments)
        .output()
        .unwrap();
    assert!(output.status.success());
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}

fn source_hash(path: &Path) -> String {
    fn visit(path: &Path, files: &mut Vec<std::path::PathBuf>) {
        for entry in fs::read_dir(path).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(&path, files);
            } else if path.extension().is_some_and(|e| e == "rs") {
                files.push(path);
            }
        }
    }
    let mut files = Vec::new();
    visit(path, &mut files);
    files.sort();
    let mut hash = Sha256::new();
    for file in files {
        hash.update(
            file.strip_prefix(path)
                .unwrap()
                .to_str()
                .unwrap()
                .as_bytes(),
        );
        hash.update([0]);
        hash.update(fs::read(file).unwrap());
        hash.update([0]);
    }
    format!("{:x}", hash.finalize())
}

#[test]
#[ignore = "release benchmark; set CONTINUGO_BENCH_SGFS and CONTINUGO_BENCH_OUTPUT"]
fn benchmark_benson_and_eyes() {
    assert!(!black_box(cfg!(debug_assertions)), "run with --release");
    let paths = env::var_os("CONTINUGO_BENCH_SGFS")
        .expect("set CONTINUGO_BENCH_SGFS (colon-separated SGF paths)");
    let output = env::var_os("CONTINUGO_BENCH_OUTPUT")
        .expect("set CONTINUGO_BENCH_OUTPUT to a new JSON file");
    let output = Path::new(&output);
    assert!(!output.exists(), "refusing to overwrite benchmark output");
    let samples = positive("CONTINUGO_BENCH_SAMPLES", 5);
    let minimum_ms = positive("CONTINUGO_BENCH_MIN_MS", 50);
    let minimum = Duration::from_millis(minimum_ms as u64);
    let full_scan = env::var("CONTINUGO_BENCH_FULL_SCAN").is_ok_and(|v| v == "1");
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let shared = root.join("../continugo");
    let mut report = json!({
        "samples": samples, "minimum_sample_ms": minimum_ms, "full_scan": full_scan,
        "allocator": "system (release libtest, not the production binary's mimalloc)",
        "train_revision": git(root, &["rev-parse", "HEAD"]),
        "train_status": git(root, &["status", "--short"]),
        "train_rust_sha256": source_hash(&root.join("src")),
        "shared_revision": git(&shared, &["rev-parse", "HEAD"]),
        "shared_status": git(&shared, &["status", "--short"]),
        "analysis_rust_sha256": source_hash(&shared.join("backend/analysis/src")),
        "rules_rust_sha256": source_hash(&shared.join("backend/rules/src")),
        "positions": [],
    });
    let save = |report: &serde_json::Value| {
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        fs::write(output, serde_json::to_vec_pretty(report).unwrap()).unwrap();
    };
    save(&report);
    let mut paths: Vec<_> = env::split_paths(&paths).collect();
    paths.sort();
    assert!(!paths.is_empty());
    for path in paths {
        let text = fs::read_to_string(&path).unwrap();
        let replay = ReplayGame::load(&text).unwrap();
        let info = replay.info(4).unwrap();
        assert!(
            info.completed && info.moves >= 10,
            "need a completed, non-smoke SGF"
        );
        let turns = info.moves - 2;
        for (phase, ply) in [
            ("early", (turns / 10).max(1)),
            ("middle", turns / 2),
            ("late", turns),
        ] {
            let game = replay.position(ply).unwrap().game;
            let started = Instant::now();
            let available = GeometricAvailability::new(game.position());
            let availability_init_ms = started.elapsed().as_secs_f64() * 1000.0;
            let benson = || {
                [Color::Black, Color::White]
                    .map(|color| unconditionally_alive(black_box(game.position()), color))
            };
            let alive = benson();
            let benson_time = Timing::new((0..samples).map(|_| sample(minimum, benson)).collect());
            let mut eye_times = Vec::new();
            let mut random_times = Vec::new();
            let mut traces = Vec::new();
            for seed in 0..samples as u64 {
                let (queries, profile) = eye_trace(&game, &available, seed);
                eye_times.push(if queries.is_empty() {
                    (0.0, 0) // Capture exemption or no legal proposals: no eye queries.
                } else {
                    sample(minimum, || eyes(&game, &available, black_box(&queries)))
                });
                random_times.push(sample(minimum, || {
                    random_move::<false>(
                        &game,
                        &available,
                        &mut StdRng::seed_from_u64(seed),
                        &mut MoveProfile::default(),
                    )
                }));
                traces.push(
                    json!({ "seed": seed, "proposals": profile.proposals, "legal": profile.legal,
                    "eye_queries": queries.len(), "rejected_eyes": profile.rejected_eyes,
                    "passed": profile.passed }),
                );
            }
            let eye_time = Timing::new(eye_times);
            let random_time = Timing::new(random_times);
            let scan = full_scan.then(|| {
                let queries: Vec<_> = [game.on_move(), game.on_move().other()]
                    .into_iter()
                    .flat_map(|color| {
                        available
                            .coordinates()
                            .iter()
                            .map(move |a| (color, a.spot().unwrap()))
                    })
                    .collect();
                let fillings = eyes(&game, &available, &queries);
                let timing = Timing::new(
                    (0..samples)
                        .map(|_| sample(minimum, || eyes(&game, &available, black_box(&queries))))
                        .collect(),
                );
                json!({"queries": queries.len(), "fillings": fillings, "timing": timing})
            });
            println!(
                "size={} {phase:6} ply={ply:3} stones={:3} free={:6} alive={:3}/{:3} benson={:.4}ms eyes={:.3}us random={:.4}ms scan={}ms",
                info.board_size,
                game.position().stones().len(),
                available.coordinates().len(),
                alive[0].iter().map(Vec::len).sum::<usize>(),
                alive[1].iter().map(Vec::len).sum::<usize>(),
                benson_time.median_ms,
                eye_time.median_ms * 1000.0,
                random_time.median_ms,
                scan.as_ref()
                    .map(|s| format!("{:.4}", s["timing"]["median_ms"].as_f64().unwrap()))
                    .unwrap_or_else(|| "off".into())
            );
            report["positions"].as_array_mut().unwrap().push(json!({
                "sgf": path, "file_sha256": format!("{:x}", Sha256::digest(text.as_bytes())),
                "game_id": info.game_id, "size": info.board_size, "phase": phase, "ply": ply,
                "stones": game.position().stones().len(), "geometric_candidates": available.coordinates().len(),
                "alive_chains": alive.each_ref().map(Vec::len),
                "alive_stones": alive.each_ref().map(|groups| groups.iter().map(Vec::len).sum::<usize>()),
                "availability_init_ms": availability_init_ms, "benson_both_colors": benson_time,
                "eye_trace": eye_time, "random_move": random_time, "traces": traces, "eye_full_scan": scan,
            }));
            save(&report);
        }
    }
    println!("benchmark results: {}", output.display());
}

#[test]
fn timing_summary_uses_the_actual_sample_median() {
    let timing = Timing::new(vec![(4.0, 1), (1.0, 4), (3.0, 2), (2.0, 3)]);
    assert_eq!(timing.median_ms, 2.5);
    assert_eq!(timing.min_ms, 1.0);
    assert_eq!(timing.max_ms, 4.0);
    assert_eq!(timing.samples_ms, [4.0, 1.0, 3.0, 2.0]);
}
