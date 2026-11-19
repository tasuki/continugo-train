use super::*;
use continugo_rules::Stone;
use rand::{Rng, SeedableRng, rngs::StdRng};

fn eye_space() -> Game {
    let mut game = Game::new(Board::new(3).unwrap());
    for y in [498, 830] {
        for x in [166, 498, 830] {
            game = game.play(Spot { x, y }).unwrap().pass().unwrap();
        }
    }
    game
}

// Independent point flood fill. No scanline components or production union-find.
fn reference(board: Board, stones: &[Stone], spot: Spot, color: Color) -> bool {
    let available = |q| board.contains(q) && stones.iter().all(|s| !board.overlaps(q, s.spot));
    if !available(spot) {
        return false;
    }
    let id = |q: Spot| (q.y * 1000 + q.x) as usize;
    let point = |i: usize| Spot {
        x: (i % 1000) as i32,
        y: (i / 1000) as i32,
    };
    let mut member = vec![false; 1_000_000];
    member[id(spot)] = true;
    let mut pending = vec![spot];
    let mut points = Vec::new();
    let mut friendly = false;
    while let Some(p) = pending.pop() {
        points.push(p);
        for dy in -1..=1 {
            for dx in -1..=1 {
                if (dx, dy) == (0, 0) {
                    continue;
                }
                let q = Spot {
                    x: p.x + dx,
                    y: p.y + dy,
                };
                if !board.contains(q) {
                    continue;
                }
                if available(q) {
                    if !member[id(q)] {
                        member[id(q)] = true;
                        pending.push(q);
                    }
                } else {
                    for stone in stones.iter().filter(|s| board.overlaps(q, s.spot)) {
                        if stone.color != color {
                            return false;
                        }
                        friendly = true;
                    }
                }
            }
        }
    }
    if !friendly {
        return false;
    }
    for p in &points {
        if board.overlaps(*p, spot) {
            member[id(*p)] = false;
        }
    }
    let mut count = 0;
    for p in points {
        if !member[id(p)] {
            continue;
        }
        count += 1;
        if count == 2 {
            return true;
        }
        member[id(p)] = false;
        pending.push(p);
        while let Some(p) = pending.pop() {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let q = Spot {
                        x: p.x + dx,
                        y: p.y + dy,
                    };
                    if board.contains(q) && member[id(q)] {
                        member[id(q)] = false;
                        pending.push(point(id(q)));
                    }
                }
            }
        }
    }
    false
}

fn benchmark_provenance() -> serde_json::Value {
    use sha2::{Digest, Sha256};
    use std::{fs, path::Path};

    fn hash_tree(root: &Path) -> String {
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
        visit(root, &mut files);
        files.sort();
        let mut hash = Sha256::new();
        for path in files {
            hash.update(
                path.strip_prefix(root)
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .as_bytes(),
            );
            hash.update([0]);
            hash.update(fs::read(path).unwrap());
            hash.update([0]);
        }
        format!("{:x}", hash.finalize())
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let git = |path: &Path, args: &[&str]| {
        let output = std::process::Command::new("git")
            .arg("-C")
            .arg(path)
            .args(args)
            .output()
            .unwrap();
        assert!(output.status.success());
        String::from_utf8(output.stdout).unwrap().trim().to_owned()
    };
    let shared = root.join("../continugo");
    serde_json::json!({
        "train_revision":git(root,&["rev-parse","HEAD"]),
        "train_status":git(root,&["status","--short"]),
        "train_rust_sha256":hash_tree(&root.join("src")),
        "shared_revision":git(&shared,&["rev-parse","HEAD"]),
        "shared_status":git(&shared,&["status","--short"]),
        "rules_rust_sha256":hash_tree(&shared.join("backend/rules/src")),
        "cargo_lock_sha256":format!("{:x}",Sha256::digest(fs::read(root.join("Cargo.lock")).unwrap()))
    })
}

#[test]
#[ignore = "release whole-game comparison; set CONTINUGO_SPLIT_SGF and CONTINUGO_SPLIT_OUTPUT"]
fn benchmark_region_splitting() {
    use crate::{
        features::{FeatureExtractor, SCHEMA_VERSION},
        replay::ReplayGame,
    };
    use serde_json::json;
    use sha2::{Digest, Sha256};
    use std::{env, fs, hint::black_box, time::Instant};
    assert!(!black_box(cfg!(debug_assertions)), "run --release");
    let source = env::var("CONTINUGO_SPLIT_SGF").unwrap();
    let output = env::var("CONTINUGO_SPLIT_OUTPUT").unwrap();
    assert!(!std::path::Path::new(&output).exists());
    let text = fs::read_to_string(&source).unwrap();
    let replay = ReplayGame::load(&text).unwrap();
    let info = replay.info(4).unwrap();
    assert!(info.completed && info.labeled);
    let repeats: usize = env::var("CONTINUGO_SPLIT_REPEATS")
        .map(|x| x.parse().unwrap())
        .unwrap_or(3);
    assert!(repeats > 0);
    let mut totals = vec![[0.0; 2]; repeats];
    let mut positions = Vec::new();
    for ply in 0..info.moves {
        let sample = replay.position(ply).unwrap();
        let mut cells: Vec<_> = sample
            .policy
            .unwrap_or_default()
            .iter()
            .filter(|p| p.visits > 0 && p.action < 1_000_000)
            .map(|p| crate::cells::cell(p.action, info.resolution))
            .collect();
        cells.sort_unstable();
        cells.dedup();
        let actions = crate::cells::actions(&cells, info.resolution);
        let mut samples = Vec::new();
        let mut positive = 0;
        let mut memory = 0;
        let mut capture_states = 0;
        for (repeat, total) in totals.iter_mut().enumerate() {
            let mut outputs: [Option<crate::features::MoveFeatures>; 2] = [None, None];
            let mut seconds = [0.0; 2];
            for step in 0..2 {
                let mode = (step + repeat + ply) % 2;
                let tick = Instant::now();
                let schema = if mode == 0 { 4 } else { SCHEMA_VERSION };
                let extractor = FeatureExtractor::for_moves(&sample.game, schema).unwrap();
                let out = extractor.moves(&actions).unwrap();
                seconds[mode] = tick.elapsed().as_secs_f64();
                black_box(&out);
                if mode == 1 {
                    positive = out
                        .exact
                        .as_chunks::<9>()
                        .0
                        .iter()
                        .filter(|r| r[4] == 1.0)
                        .count();
                    assert!(
                        out.exact
                            .as_chunks::<9>()
                            .0
                            .iter()
                            .all(|r| r[4] == 0.0 || r[4] == 1.0)
                    );
                    let cache = extractor.splitting.captures.lock().unwrap();
                    memory =
                        extractor.splitting.parent.get().map_or(0, |i| i.bytes()) + cache.bytes;
                    capture_states = cache.entries.len();
                }
                outputs[mode] = Some(out);
                total[mode] += seconds[mode];
            }
            let old = outputs[0].as_ref().unwrap();
            let new = outputs[1].as_ref().unwrap();
            assert_eq!(old.legal, new.legal);
            for (a, b) in old.exact.chunks_exact(10).zip(new.exact.as_chunks::<9>().0) {
                for i in 0..4 {
                    let count =
                        (f64::from(a[i + 1]) * 1_000_001_f64.ln()).exp_m1().round() as usize;
                    assert_eq!(crate::features::count_feature(count), b[i]);
                }
                assert!(
                    a[6..]
                        .iter()
                        .zip(&b[5..])
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
            samples.push(seconds);
        }
        positions.push(json!({"ply":ply,"coordinates":actions.len(),"positive":positive,"index_bytes":memory,"capture_states":capture_states,"seconds":samples}));
    }
    let report = json!({"source":source,"sgf_sha256":format!("{:x}",Sha256::digest(text.as_bytes())),
        "info":info,"repeats":repeats,"modes":["rules_counts_borders_only","friendly_region_splitting"],
        "allocator":"release libtest system allocator; serial; full fine extraction including preparation",
        "provenance":benchmark_provenance(),"totals_seconds":totals,"positions":positions});
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
fn rare_dividing_moves_not_smoothed_and_colour_matters() {
    let game = eye_space();
    let index = Index::new(game.board(), game.position().stones());
    for x in 495..=505 {
        let p = Spot { x, y: 166 };
        let expected = (498..=502).contains(&x);
        assert_eq!(index.splits(p, Color::Black), expected, "x={x}");
        assert_eq!(
            index.splits(p, Color::Black),
            reference(game.board(), game.position().stones(), p, Color::Black)
        );
        assert!(!index.splits(p, Color::White));
    }
    let mut mixed = game.position().stones().to_vec();
    mixed[0].color = Color::White;
    let index = Index::new(game.board(), &mixed);
    assert!(!index.splits(Spot { x: 498, y: 166 }, Color::Black));
    assert_eq!(
        index.splits(Spot { x: 498, y: 166 }, Color::Black),
        reference(game.board(), &mixed, Spot { x: 498, y: 166 }, Color::Black)
    );
}

#[test]
fn exact_one_coordinate_split_and_eight_neighbour_connection() {
    let d: i32 = 300;
    let widths: Vec<_> = (0..d).map(|y| (d * d - 1 - y * y).isqrt()).collect();
    let rows = [RowRun {
        y: 500,
        left: 200,
        right: 800,
    }];
    let mut scratch = Scratch::default();
    for x in 499..=501 {
        assert_eq!(scratch.splits(&rows, Spot { x, y: 500 }, &widths), x == 500);
    }
    // A diagonal one-coordinate passage is connected, not two eyes.
    let rows = [
        RowRun {
            y: 400,
            left: 400,
            right: 400,
        },
        RowRun {
            y: 401,
            left: 401,
            right: 401,
        },
    ];
    assert!(!scratch.splits(&rows, Spot { x: 900, y: 900 }, &widths));
}

#[test]
fn cache_uses_actual_survivors_including_captures() {
    let game = eye_space();
    let cache = Cache::default();
    for x in [497, 498, 499, 502, 503] {
        let spot = Spot { x, y: 166 };
        let child = game.prepare().play(spot).unwrap();
        let survivors = &child.position().stones()[..child.position().stones().len() - 1];
        assert_eq!(
            cache.value(&game, spot, child.position()) != 0.0,
            reference(game.board(), survivors, spot, game.on_move())
        );
    }
    let corner = Game::new(Board::new(9).unwrap())
        .play(Spot { x: 55, y: 55 })
        .unwrap();
    let prepared = corner.position().prepare();
    let spot = prepared
        .analyze_group(prepared.groups(Color::Black).next().unwrap())
        .capture_in_one
        .unwrap();
    let child = corner.play(spot).unwrap();
    assert_eq!(child.position().stones().len(), 1);
    assert_eq!(cache.value(&corner, spot, child.position()), 0.0);

    // An enemy in the left pocket is captured by the dividing move. Checking
    // fixed pre-move obstacles would incorrectly reject the friendly boundary.
    let original = eye_space();
    let record = original.record();
    let invaded = Game::from_record_on(original.board(), &record[..record.len() - 1])
        .unwrap()
        .play(Spot { x: 166, y: 166 })
        .unwrap();
    let spot = Spot { x: 498, y: 166 };
    let analysis = invaded
        .position()
        .prepare()
        .analyze_move(
            Stone {
                color: invaded.on_move(),
                spot,
            },
            0,
        )
        .unwrap();
    assert_eq!(analysis.captured_stones, 1);
    assert!(
        !Index::new(invaded.board(), invaded.position().stones()).splits(spot, invaded.on_move())
    );
    assert_eq!(
        Cache::default().value(&invaded, spot, &analysis.position),
        1.0
    );
    let actions: Vec<_> = (495..=505)
        .map(|x| continugo_rules::Action::from_index(166000 + x).unwrap())
        .collect();
    let actions = actions.repeat(400); // More than one 4096-row worker chunk.
    let extractor =
        crate::features::FeatureExtractor::for_moves(&invaded, crate::features::SCHEMA_VERSION)
            .unwrap();
    let serial = extractor.moves(&actions).unwrap();
    let extractor =
        crate::features::FeatureExtractor::for_moves(&invaded, crate::features::SCHEMA_VERSION)
            .unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap();
    let parallel = extractor.moves_parallel(&actions, &pool).unwrap();
    assert_eq!(serial.exact, parallel.exact);
    assert_eq!(serial.legal, parallel.legal);
}

#[test]
fn scanline_matches_independent_flood_fill_on_dense_mixed_positions() {
    let mut rng = StdRng::seed_from_u64(20261108);
    for size in [3, 5, 9] {
        let board = Board::new(size).unwrap();
        let mut game = Game::new(board);
        for turn in 0..70 {
            for _ in 0..1000 {
                let p = Spot {
                    x: rng.random_range(0..1000),
                    y: rng.random_range(0..1000),
                };
                if let Ok(next) = game.play(p) {
                    game = next;
                    break;
                }
            }
            if turn % 10 != 9 {
                continue;
            }
            let cache = Cache::default();
            for _ in 0..12 {
                let p = Spot {
                    x: rng.random_range(0..1000),
                    y: rng.random_range(0..1000),
                };
                let Ok(child) = game.play(p) else {
                    continue;
                };
                let survivors = &child.position().stones()[..child.position().stones().len() - 1];
                assert_eq!(
                    cache.value(&game, p, child.position()) != 0.0,
                    reference(board, survivors, p, game.on_move()),
                    "size={size} turn={turn} p={p:?}"
                );
            }
            let captures = cache.captures.lock().unwrap();
            assert!(captures.entries.len() <= CAPTURE_STATES);
            assert!(captures.bytes <= CAPTURE_BYTES);
        }
    }
}
