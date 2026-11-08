//! Historical footprint-contract experiments, not current schema-5 fine inputs.
//! Test-only experiments: avoid full child construction on certified ordinary
//! moves, and replace repeated exact projection divisions with a small table.
//! Any rules fast path belongs in shared rules before production adoption.
use super::*;
use crate::replay::ReplayGame;
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{env, fs, hint::black_box, path::Path};

struct FastState {
    friendly: Vec<Vec<usize>>,
    // Two known liberties per group. Groups with witnesses >= 2*d apart cannot
    // lose both to any one strict diameter exclusion disc and need no check.
    vulnerable: Vec<[Spot; 2]>,
}

#[derive(Clone, Copy, Debug)]
enum Decision {
    Illegal,
    Legal { groups: usize, stones: usize },
    Fallback,
}

impl FastState {
    fn new(extractor: &FeatureExtractor<'_>) -> Self {
        let game = extractor.game;
        let d = i64::from(game.board().diameter());
        let mut vulnerable = Vec::new();
        for color in [game.on_move(), game.on_move().other()] {
            for members in extractor.prepared.groups(color) {
                let spans = extractor.prepared.liberty_spans(members);
                assert!(!spans.is_empty());
                let points: Vec<_> = spans
                    .iter()
                    .flat_map(|s| [Spot { x: s.left, y: s.y }, Spot { x: s.right, y: s.y }])
                    .collect();
                let mut extrema = Vec::new();
                for axis in 0..2 {
                    let coordinate = |s: &&Spot| if axis == 0 { s.x } else { s.y };
                    extrema.push(**points.iter().min_by_key(coordinate).as_ref().unwrap());
                    extrema.push(**points.iter().max_by_key(coordinate).as_ref().unwrap());
                }
                let mut pair = [points[0]; 2];
                let mut separation = 0;
                for &a in &extrema {
                    for &b in &extrema {
                        let distance = i64::from(a.x - b.x).pow(2) + i64::from(a.y - b.y).pow(2);
                        if distance > separation {
                            pair = [a, b];
                            separation = distance;
                        }
                    }
                }
                if separation < 4 * d * d {
                    vulnerable.push(pair);
                }
            }
        }
        Self {
            friendly: extractor
                .prepared
                .groups(game.on_move())
                .map(<[usize]>::to_vec)
                .collect(),
            vulnerable,
        }
    }

    fn decide(&self, extractor: &FeatureExtractor<'_>, spot: Spot) -> Decision {
        let game = extractor.game;
        let board = game.board();
        if !game.position().can_place(spot) {
            return Decision::Illegal;
        }
        // This only certifies no old group dies; failure is not a capture claim.
        if self
            .vulnerable
            .iter()
            .any(|w| w.iter().all(|&p| board.overlaps(p, spot)))
        {
            return Decision::Fallback;
        }
        let stones = game.position().stones();
        let (mut groups, mut count) = (0, 0);
        for members in &self.friendly {
            if members
                .iter()
                .any(|&i| board.adjacent(stones[i].spot, spot))
            {
                groups += 1;
                count += members.len();
            }
        }
        if groups != 0 {
            return Decision::Legal {
                groups,
                stones: count,
            };
        }
        // An unjoined new stone needs its own surviving liberty. Four exact
        // points are sufficient witnesses, not an exhaustive legality test.
        let d = board.diameter();
        for (dx, dy) in [(d, 0), (-d, 0), (0, d), (0, -d)] {
            if extractor.prepared.is_local_liberty_center(
                spot,
                Spot {
                    x: spot.x + dx,
                    y: spot.y + dy,
                },
            ) {
                return Decision::Legal {
                    groups: 0,
                    stones: 0,
                };
            }
        }
        Decision::Fallback
    }
}

#[derive(Default, Serialize)]
struct Counts {
    coordinates: usize,
    illegal_geometry: usize,
    certified_legal: usize,
    fallback: usize,
    groups_requiring_witness_checks: usize,
}

#[derive(Default, Serialize)]
struct Components {
    rules_preparation: crate::profile::Stage,
    extra_preparation: crate::profile::Stage,
    allocation: crate::profile::Stage,
    borders: crate::profile::Stage,
    rules_and_counts: crate::profile::Stage,
    local_footprint: crate::profile::Stage,
    footprint: footprint::profile::Profile,
}

fn experiment(
    extractor: &FeatureExtractor<'_>,
    actions: &[Action],
    rules: bool,
    lookup: bool,
) -> (MoveFeatures, Counts) {
    experiment_inner::<false>(
        extractor,
        actions,
        rules,
        lookup,
        &mut Components::default(),
    )
}

fn experiment_inner<const PROFILE: bool>(
    extractor: &FeatureExtractor<'_>,
    actions: &[Action],
    rules: bool,
    lookup: bool,
    profile: &mut Components,
) -> (MoveFeatures, Counts) {
    let tick = PROFILE.then(Instant::now);
    let state = rules.then(|| FastState::new(extractor));
    let projection =
        lookup.then(|| footprint::profile::Projection::new(extractor.game.board().stone_radius()));
    if let Some(tick) = tick {
        profile.extra_preparation.record(tick);
    }
    let mut counts = Counts {
        coordinates: actions.len(),
        groups_requiring_witness_checks: state.as_ref().map_or(0, |s| s.vulnerable.len()),
        ..Default::default()
    };
    let tick = PROFILE.then(Instant::now);
    let mut out = MoveFeatures {
        exact: vec![0.0; actions.len() * 11],
        legal: vec![0; actions.len()],
    };
    if let Some(tick) = tick {
        profile.allocation.record(tick);
    }
    for ((action, row), legal) in actions
        .iter()
        .zip(out.exact.chunks_mut(11))
        .zip(&mut out.legal)
    {
        let spot = action.spot().unwrap();
        let tick = PROFILE.then(Instant::now);
        row[7..].copy_from_slice(&extractor.borders(spot));
        if let Some(tick) = tick {
            profile.borders.record(tick);
        }
        let tick = PROFILE.then(Instant::now);
        let decision = state
            .as_ref()
            .map_or(Decision::Fallback, |s| s.decide(extractor, spot));
        match decision {
            Decision::Illegal => {
                counts.illegal_geometry += 1;
                if let Some(tick) = tick {
                    profile.rules_and_counts.record(tick);
                }
                continue;
            }
            Decision::Legal { groups, stones } => {
                counts.certified_legal += 1;
                row[..6].copy_from_slice(&[
                    1.0,
                    count_feature(groups),
                    count_feature(stones),
                    0.0,
                    0.0,
                    count_feature(stones + 1),
                ]);
            }
            Decision::Fallback => {
                counts.fallback += 1;
                let Some(analysis) = extractor.prepared.analyze_move(
                    Stone {
                        color: extractor.game.on_move(),
                        spot,
                    },
                    0,
                ) else {
                    if let Some(tick) = tick {
                        profile.rules_and_counts.record(tick);
                    }
                    continue;
                };
                row[..6].copy_from_slice(&[
                    1.0,
                    count_feature(analysis.connected_groups),
                    count_feature(analysis.connected_stones),
                    count_feature(analysis.captured_groups),
                    count_feature(analysis.captured_stones),
                    count_feature(analysis.group_size),
                ]);
            }
        }
        *legal = 1;
        if let Some(tick) = tick {
            profile.rules_and_counts.record(tick);
        }
        let tick = PROFILE.then(Instant::now);
        row[6] = if PROFILE {
            if let Some(projection) = &projection {
                footprint::profile::area_lookup_profiled(
                    &extractor.prepared,
                    spot,
                    extractor.game.board().stone_radius(),
                    &extractor.footprints,
                    projection,
                    &mut profile.footprint,
                )
            } else {
                footprint::profile::area(
                    &extractor.prepared,
                    spot,
                    extractor.game.board().stone_radius(),
                    &extractor.footprints,
                    &mut profile.footprint,
                )
            }
        } else if let Some(projection) = &projection {
            footprint::profile::area_lookup(
                &extractor.prepared,
                spot,
                extractor.game.board().stone_radius(),
                &extractor.footprints,
                projection,
            )
        } else {
            footprint::area(
                &extractor.prepared,
                spot,
                extractor.game.board().stone_radius(),
                &extractor.footprints,
            )
        };
        if let Some(tick) = tick {
            profile.local_footprint.record(tick);
        }
    }
    (out, counts)
}

#[test]
fn certificates_and_projection_preserve_fine_bits() {
    use continugo_rules::Board;
    let corner = Game::new(Board::new(9).unwrap())
        .play(Spot { x: 55, y: 55 })
        .unwrap();
    let witness = corner
        .position()
        .prepare()
        .analyze_group(&corner.position().groups(corner.on_move().other())[0])
        .capture_in_one
        .unwrap();
    let corner_extractor = FeatureExtractor::for_moves(&corner, SCHEMA_VERSION).unwrap();
    assert!(matches!(
        FastState::new(&corner_extractor).decide(&corner_extractor, witness),
        Decision::Fallback
    ));
    let captured = corner.play(witness).unwrap();
    let merge = Game::new(Board::new(9).unwrap())
        .play(Spot { x: 300, y: 500 })
        .unwrap()
        .pass()
        .unwrap()
        .play(Spot { x: 550, y: 500 })
        .unwrap()
        .pass()
        .unwrap();
    let mut actions: Vec<_> = (0..COORDINATE_ACTIONS)
        .step_by(251)
        .map(|id| Action::from_index(id).unwrap())
        .collect();
    actions.extend([
        Action::from_index((witness.x + 1000 * witness.y) as u32).unwrap(),
        Action::from_index(500425).unwrap(),
    ]);
    for game in [Game::new(Board::new(9).unwrap()), corner, captured, merge] {
        let extractor = FeatureExtractor::for_moves(&game, SCHEMA_VERSION).unwrap();
        let expected = extractor.moves_footprint(&actions).unwrap();
        for (rules, lookup) in [(true, false), (false, true), (true, true)] {
            let extractor = FeatureExtractor::for_moves(&game, SCHEMA_VERSION).unwrap();
            let (actual, _) = experiment(&extractor, &actions, rules, lookup);
            super::cache_bench::bits_equal(&expected.exact, &actual.exact);
            assert_eq!(expected.legal, actual.legal);
        }
    }
    // Every supported radius/phase, including endpoints and isolated centers.
    for radius in 1..=500 {
        let projection = footprint::profile::Projection::new(radius);
        projection.check(radius);
    }
}

#[test]
fn shortcuts_match_dense_positions_and_both_colors_across_sizes() {
    use continugo_rules::Board;
    use rand::{Rng, SeedableRng, rngs::StdRng};
    let mut rng = StdRng::seed_from_u64(20261108);
    for size in [9, 13, 19] {
        let mut game = Game::new(Board::new(size).unwrap());
        for turn in 0..90 {
            for _ in 0..2000 {
                let spot = Spot {
                    x: rng.random_range(0..1000),
                    y: rng.random_range(0..1000),
                };
                if let Ok(next) = game.play(spot) {
                    game = next;
                    break;
                }
            }
            if turn % 9 != 0 {
                continue;
            }
            let actions: Vec<_> = (0..300)
                .map(|_| Action::from_index(rng.random_range(0..COORDINATE_ACTIONS)).unwrap())
                .collect();
            let extractor = FeatureExtractor::for_moves(&game, SCHEMA_VERSION).unwrap();
            let expected = extractor.moves_footprint(&actions).unwrap();
            let extractor = FeatureExtractor::for_moves(&game, SCHEMA_VERSION).unwrap();
            let (actual, _) = experiment(&extractor, &actions, true, true);
            super::cache_bench::bits_equal(&expected.exact, &actual.exact);
            assert_eq!(expected.legal, actual.legal);
        }
    }
}

#[test]
#[ignore = "paired component clocks; set CONTINUGO_FINE_SGF and CONTINUGO_FINE_OUTPUT"]
fn profile_fine_shortcut_components() {
    assert!(!black_box(cfg!(debug_assertions)), "run --release");
    let source = env::var("CONTINUGO_FINE_SGF").unwrap();
    let output = env::var("CONTINUGO_FINE_OUTPUT").unwrap();
    assert!(!Path::new(&output).exists());
    let text = fs::read_to_string(&source).unwrap();
    let replay = ReplayGame::load(&text).unwrap();
    let info = replay.info(4).unwrap();
    assert!(info.completed && info.labeled && !info.eligible_plies.is_empty());
    let repeats: usize = env::var("CONTINUGO_FINE_REPEATS")
        .map(|x| x.parse().unwrap())
        .unwrap_or(3);
    assert!(repeats > 0);
    let mut positions = Vec::new();
    for ply in 0..info.moves {
        let sample = replay.position(ply).unwrap();
        let mut cells: Vec<_> = sample
            .policy
            .unwrap_or_default()
            .iter()
            .filter(|p| p.visits > 0 && p.action < COORDINATE_ACTIONS)
            .map(|p| crate::cells::cell(p.action, info.resolution))
            .collect();
        cells.sort_unstable();
        cells.dedup();
        let actions = crate::cells::actions(&cells, info.resolution);
        let expected = FeatureExtractor::for_moves(&sample.game, SCHEMA_VERSION)
            .unwrap()
            .moves_footprint(&actions)
            .unwrap();
        let mut samples = Vec::new();
        for repeat in 0..repeats {
            for step in 0..2 {
                let now = (ply + repeat + step) % 2 != 0;
                let mut components = Components::default();
                let total = Instant::now();
                let started = Instant::now();
                let extractor = FeatureExtractor::for_moves(&sample.game, SCHEMA_VERSION).unwrap();
                components.rules_preparation.record(started);
                let (actual, counts) =
                    experiment_inner::<true>(&extractor, &actions, now, now, &mut components);
                let seconds = total.elapsed().as_secs_f64();
                super::cache_bench::bits_equal(&expected.exact, &actual.exact);
                assert_eq!(expected.legal, actual.legal);
                samples.push(json!({"repeat":repeat,"now":now,"seconds":seconds,"components":components,"counts":counts}));
            }
        }
        positions.push(json!({"ply":ply,"coordinates":actions.len(),"samples":samples}));
    }
    let report = json!({"source":source,"sgf_sha256":format!("{:x}",Sha256::digest(text.as_bytes())),
        "info":info,"repeats":repeats,"provenance":super::cache_bench::provenance(),
        "allocator":"release libtest system allocator; serial","positions":positions});
    use std::io::Write;
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap()
        .write_all(&serde_json::to_vec_pretty(&report).unwrap())
        .unwrap();
}

#[test]
#[ignore = "whole-game fine feature experiment; set CONTINUGO_FINE_SGF and CONTINUGO_FINE_OUTPUT"]
fn profile_fine_shortcuts() {
    assert!(!black_box(cfg!(debug_assertions)), "run --release");
    let source = env::var("CONTINUGO_FINE_SGF").unwrap();
    let output = env::var("CONTINUGO_FINE_OUTPUT").unwrap();
    assert!(!Path::new(&output).exists());
    let text = fs::read_to_string(&source).unwrap();
    let replay = ReplayGame::load(&text).unwrap();
    let info = replay.info(4).unwrap();
    assert!(info.completed && info.labeled && !info.eligible_plies.is_empty());
    let repeats: usize = env::var("CONTINUGO_FINE_REPEATS")
        .map(|x| x.parse().unwrap())
        .unwrap_or(3);
    assert!(repeats > 0);
    let mut positions = Vec::new();
    let mut totals = vec![[0.0; 4]; repeats];
    for ply in 0..info.moves {
        let sample = replay.position(ply).unwrap();
        let mut cells: Vec<_> = sample
            .policy
            .unwrap_or_default()
            .iter()
            .filter(|p| p.visits > 0 && p.action < COORDINATE_ACTIONS)
            .map(|p| crate::cells::cell(p.action, info.resolution))
            .collect();
        cells.sort_unstable();
        cells.dedup();
        let actions = crate::cells::actions(&cells, info.resolution);
        let expected = FeatureExtractor::for_moves(&sample.game, SCHEMA_VERSION)
            .unwrap()
            .moves_footprint(&actions)
            .unwrap();
        let mut times = Vec::new();
        let mut counts = None;
        for (repeat, total) in totals.iter_mut().enumerate() {
            let mut seconds = [0.0; 4];
            // Rotate the four methods, so baseline is not always measured first.
            for step in 0..4 {
                let mode = (step + repeat + ply) % 4;
                let started = Instant::now();
                let extractor = FeatureExtractor::for_moves(&sample.game, SCHEMA_VERSION).unwrap();
                let actual = if mode == 0 {
                    extractor.moves_footprint(&actions).unwrap()
                } else {
                    let (actual, c) = experiment(
                        &extractor,
                        &actions,
                        mode == 1 || mode == 3,
                        mode == 2 || mode == 3,
                    );
                    if mode == 1 {
                        counts = Some(c);
                    }
                    actual
                };
                seconds[mode] = started.elapsed().as_secs_f64();
                black_box(&actual);
                super::cache_bench::bits_equal(&expected.exact, &actual.exact);
                assert_eq!(expected.legal, actual.legal, "ply={ply}, mode={mode}");
                total[mode] += seconds[mode];
            }
            times.push(seconds);
        }
        positions
            .push(json!({"ply":ply,"coordinates":actions.len(),"counts":counts,"seconds":times}));
    }
    let report = json!({"source":source,"sgf_sha256":format!("{:x}",Sha256::digest(text.as_bytes())),
        "info":info,"provenance":super::cache_bench::provenance(),"repeats":repeats,
        "modes":["baseline","no_capture_certificate","projection_lookup","both"],
        "allocator":"release libtest system allocator; serial",
        "totals_seconds":totals,"positions":positions});
    use std::io::Write;
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap()
        .write_all(&serde_json::to_vec_pretty(&report).unwrap())
        .unwrap();
}
