//! Opt-in whole-game feature cost and one-edge reuse audit. No production cache.
//! Historical footprint contract only; not current schema-5 fine inputs.
//! Uses real positive-visit target cells, not an invented policy or full domain.
use super::*;
use crate::replay::ReplayGame;
use continugo_rules::{Color, LibertySpan};
use serde::Serialize;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, env, fs, hint::black_box, path::Path, time::Instant};

#[derive(Default, Serialize)]
struct FineProfile {
    allocation: Stage,
    borders: Stage,
    legal_analysis: Stage,
    illegal_analysis: Stage,
    local_footprint: Stage,
    /// Subset of local_footprint, not an additional disjoint stage.
    parent_reusable_footprint: Stage,
    /// Subset of local_footprint; simulated bounded inheritance along this line.
    ancestor_reusable_footprint: Stage,
    /// Subdivision of local_footprint.
    footprint: footprint::profile::Profile,
    fields: Stage,
    analysis_drop: Stage,
}

// Diagnostic decomposition of the serial production loop. Every result is
// checked bitwise against ordinary moves(); this is not a second rules engine.
fn fine_profile(
    extractor: &FeatureExtractor<'_>,
    actions: &[Action],
    reusable: &[bool],
    inherited: &[Option<f32>],
) -> (MoveFeatures, FineProfile) {
    let mut p = FineProfile::default();
    let tick = Instant::now();
    let mut out = MoveFeatures {
        exact: vec![0.0; actions.len() * MOVE_FEATURES.len()],
        legal: vec![0; actions.len()],
    };
    p.allocation.record(tick);
    for (i, ((action, row), legal)) in actions
        .iter()
        .zip(out.exact.chunks_mut(11))
        .zip(&mut out.legal)
        .enumerate()
    {
        let spot = action.spot().unwrap();
        let tick = Instant::now();
        row[7..].copy_from_slice(&extractor.borders(spot));
        p.borders.record(tick);
        let tick = Instant::now();
        let analysis = extractor.prepared.analyze_move(
            Stone {
                color: extractor.game.on_move(),
                spot,
            },
            0,
        );
        let Some(analysis) = analysis else {
            p.illegal_analysis.record(tick);
            continue;
        };
        p.legal_analysis.record(tick);
        let tick = Instant::now();
        *legal = 1;
        row[..6].copy_from_slice(&[
            1.0,
            count_feature(analysis.connected_groups),
            count_feature(analysis.connected_stones),
            count_feature(analysis.captured_groups),
            count_feature(analysis.captured_stones),
            count_feature(analysis.group_size),
        ]);
        p.fields.record(tick);
        let tick = Instant::now();
        row[6] = footprint::profile::area(
            &extractor.prepared,
            spot,
            extractor.game.board().stone_radius(),
            &extractor.footprints,
            &mut p.footprint,
        );
        let seconds = p.local_footprint.record(tick);
        if reusable[i] {
            p.parent_reusable_footprint.calls += 1;
            p.parent_reusable_footprint.seconds += seconds;
        }
        if let Some(value) = inherited[i] {
            assert_eq!(
                value.to_bits(),
                row[6].to_bits(),
                "ancestor footprint invalidation failed"
            );
            p.ancestor_reusable_footprint.calls += 1;
            p.ancestor_reusable_footprint.seconds += seconds;
        }
        let tick = Instant::now();
        drop(analysis);
        p.analysis_drop.record(tick);
    }
    (out, p)
}

pub(super) fn bits_equal(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
}

// Stable geometry identities: child captures compact stone indices, so source
// member indices cannot be used as cross-position cache keys.
type GroupKey = Vec<[i32; 3]>;
struct GroupSnapshot {
    spans: Vec<LibertySpan>,
}
fn groups(extractor: &FeatureExtractor<'_>) -> BTreeMap<GroupKey, GroupSnapshot> {
    let stones = extractor.game.position().stones();
    let mut result = BTreeMap::new();
    for color in [Color::Black, Color::White] {
        for members in extractor.prepared.groups(color) {
            let mut key: Vec<_> = members
                .iter()
                .map(|&i| {
                    let s = stones[i];
                    [i32::from(s.color == Color::White), s.spot.x, s.spot.y]
                })
                .collect();
            key.sort_unstable();
            result.insert(
                key,
                GroupSnapshot {
                    spans: extractor.prepared.liberty_spans(members),
                },
            );
        }
    }
    result
}

fn changed_stones(parent: &Game, child: &Game) -> Vec<Stone> {
    let a = parent.position().stones();
    let b = child.position().stones();
    a.iter()
        .filter(|s| !b.contains(s))
        .chain(b.iter().filter(|s| !a.contains(s)))
        .copied()
        .collect()
}

// A changed obstacle can affect a local annulus only within (sqrt(2)+1)*d.
// Round upward, and mark the boundary dirty too: conservative, integer checks.
fn influence(diameter: i32) -> i64 {
    (f64::from(diameter) * (2.0f64.sqrt() + 1.0)).ceil() as i64
}
fn safe(spot: Spot, changes: &[Stone], diameter: i32) -> bool {
    let reach = influence(diameter);
    changes.iter().all(|s| {
        i64::from(spot.x - s.spot.x).pow(2) + i64::from(spot.y - s.spot.y).pow(2) > reach * reach
    })
}

#[derive(Default, Serialize)]
struct EdgeAudit {
    pass: bool,
    removed_stones: usize,
    merged_parent_groups: usize,
    child_groups: usize,
    same_members: usize,
    same_members_and_spans: usize,
    guaranteed_clean_groups: usize,
    blocker_rows: usize,
    guaranteed_clean_blocker_rows: usize,
    // Occupancy/group size/centers/footprints, both colours, excluding borders.
    raster_values: usize,
    unchanged_raster_values: [usize; 4],
    child_coordinates: usize,
    requested_in_parent: usize,
    legal_in_both: usize,
    reusable_local_footprints: usize,
    equal_local_footprints: usize,
    retained_parent_payload_bytes: usize,
}

struct Snapshot {
    game: Game,
    coarse: Vec<f32>,
    groups: BTreeMap<GroupKey, GroupSnapshot>,
    actions: Vec<Action>,
    moves: MoveFeatures,
}

// Metadata-only cache simulation: extraction still computes every value. Exact
// coordinate tags reject collisions. At most 65,536 slots, no unbounded history.
const LOCAL_SLOTS: usize = 65_536;
struct LocalCache(Vec<Option<(u32, f32)>>);
impl LocalCache {
    fn new() -> Self {
        Self(vec![None; LOCAL_SLOTS])
    }
    fn slot(id: u32) -> usize {
        (u64::from(id).wrapping_mul(0x9e3779b185ebca87) >> (64 - LOCAL_SLOTS.ilog2())) as usize
    }
    fn invalidate(&mut self, changes: &[Stone], diameter: i32) {
        for entry in &mut self.0 {
            if entry.is_some_and(|(id, _)| {
                !safe(
                    Action::from_index(id).unwrap().spot().unwrap(),
                    changes,
                    diameter,
                )
            }) {
                *entry = None;
            }
        }
    }
    fn get(&self, id: u32) -> Option<f32> {
        self.0[Self::slot(id)]
            .filter(|(tag, _)| *tag == id)
            .map(|(_, v)| v)
    }
    fn insert(&mut self, id: u32, value: f32) {
        self.0[Self::slot(id)] = Some((id, value));
    }
    fn len(&self) -> usize {
        self.0.iter().filter(|e| e.is_some()).count()
    }
}

fn reusable_local_mask(parent: &Snapshot, child: &Game, actions: &[Action]) -> Vec<bool> {
    let changes = changed_stones(&parent.game, child);
    let old: BTreeMap<_, _> = parent
        .actions
        .iter()
        .enumerate()
        .map(|(i, a)| (a.index(), i))
        .collect();
    actions
        .iter()
        .map(|a| {
            old.get(&a.index())
                .is_some_and(|&i| parent.moves.legal[i] != 0)
                && safe(a.spot().unwrap(), &changes, child.board().diameter())
        })
        .collect()
}

fn audit(parent: &Snapshot, child: &Snapshot) -> EdgeAudit {
    let changes = changed_stones(&parent.game, &child.game);
    let board = child.game.board();
    let mut out = EdgeAudit {
        pass: changes.is_empty(),
        removed_stones: parent
            .game
            .position()
            .stones()
            .iter()
            .filter(|s| !child.game.position().stones().contains(s))
            .count(),
        child_groups: child.groups.len(),
        child_coordinates: child.actions.len(),
        retained_parent_payload_bytes: parent.coarse.len() * 4
            + parent.actions.len() * (4 + 1 + 11 * 4)
            + parent
                .groups
                .iter()
                .map(|(key, value)| key.len() * 12 + value.spans.len() * 12)
                .sum::<usize>(),
        ..Default::default()
    };
    // Count old friendly groups that participate in a newly enlarged/merged group.
    for key in child
        .groups
        .keys()
        .filter(|key| !parent.groups.contains_key(*key))
    {
        let joined = parent
            .groups
            .keys()
            .filter(|old| old.iter().all(|s| key.contains(s)))
            .count();
        if joined > 1 {
            out.merged_parent_groups += joined;
        }
    }
    for (key, group) in &child.groups {
        if let Some(old) = parent.groups.get(key) {
            out.same_members += 1;
            out.same_members_and_spans += usize::from(old.spans == group.spans);
            if key
                .iter()
                .all(|s| safe(Spot { x: s[1], y: s[2] }, &changes, board.diameter()))
            {
                out.guaranteed_clean_groups += 1;
                assert_eq!(old.spans, group.spans, "clean-group invalidation failed");
            }
        }
    }
    let low = board.stone_radius();
    let high = 1000 - low;
    out.blocker_rows = (high - low + 1) as usize;
    out.guaranteed_clean_blocker_rows = (low..=high)
        .filter(|&y| {
            changes
                .iter()
                .all(|s| (y - s.spot.y).abs() >= board.diameter())
        })
        .count();
    let area = child.coarse.len() / 12;
    out.raster_values = 2 * area;
    for (kind, same) in out.unchanged_raster_values.iter_mut().enumerate() {
        for color in 0..2 {
            // One ply flips all friendly/enemy pairs, even a pass.
            let a = (2 * kind + color) * area;
            let b = (2 * kind + 1 - color) * area;
            *same += parent.coarse[a..a + area]
                .iter()
                .zip(&child.coarse[b..b + area])
                .filter(|(a, b)| a.to_bits() == b.to_bits())
                .count();
        }
    }
    bits_equal(&parent.coarse[8 * area..], &child.coarse[8 * area..]);
    let old: BTreeMap<_, _> = parent
        .actions
        .iter()
        .enumerate()
        .map(|(i, a)| (a.index(), i))
        .collect();
    for (j, action) in child.actions.iter().enumerate() {
        let Some(&i) = old.get(&action.index()) else {
            continue;
        };
        out.requested_in_parent += 1;
        if parent.moves.legal[i] == 0 || child.moves.legal[j] == 0 {
            continue;
        }
        out.legal_in_both += 1;
        let a = parent.moves.exact[i * 11 + 6].to_bits();
        let b = child.moves.exact[j * 11 + 6].to_bits();
        out.equal_local_footprints += usize::from(a == b);
        if safe(action.spot().unwrap(), &changes, board.diameter()) {
            out.reusable_local_footprints += 1;
            assert_eq!(a, b, "local-footprint invalidation failed");
        }
    }
    out
}

fn cells(replay: &ReplayGame, ply: usize, resolution: usize) -> Vec<usize> {
    let mut cells: Vec<_> = replay
        .position(ply)
        .unwrap()
        .policy
        .unwrap_or_default()
        .iter()
        .filter(|t| t.visits > 0 && t.action < COORDINATE_ACTIONS)
        .map(|t| crate::cells::cell(t.action, resolution))
        .collect();
    cells.sort_unstable();
    cells.dedup();
    cells
}

fn positive(name: &str, default: usize) -> usize {
    let n = env::var(name)
        .map(|v| v.parse().unwrap())
        .unwrap_or(default);
    assert!(n > 0);
    n
}

pub(super) fn provenance() -> serde_json::Value {
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
    json!({"train_revision":git(root,&["rev-parse","HEAD"]), "train_status":git(root,&["status","--short"]),
        "train_rust_sha256":hash_tree(&root.join("src")),
        "shared_revision":git(&shared,&["rev-parse","HEAD"]), "shared_status":git(&shared,&["status","--short"]),
        "rules_rust_sha256":hash_tree(&shared.join("backend/rules/src")),
        "cargo_lock_sha256":format!("{:x}",Sha256::digest(fs::read(root.join("Cargo.lock")).unwrap()))})
}

#[test]
#[ignore = "whole-game release audit; set CONTINUGO_CACHE_SGF and CONTINUGO_CACHE_OUTPUT"]
fn profile_parent_child_features() {
    assert!(!black_box(cfg!(debug_assertions)), "run --release");
    let path = env::var("CONTINUGO_CACHE_SGF").unwrap();
    let output = env::var("CONTINUGO_CACHE_OUTPUT").unwrap();
    assert!(!Path::new(&output).exists(), "refusing to overwrite output");
    let text = fs::read_to_string(&path).unwrap();
    let replay = ReplayGame::load(&text).unwrap();
    let info = replay.info(4).unwrap();
    assert!(info.completed && info.labeled && !info.eligible_plies.is_empty());
    let provenance = provenance();
    let repeats = positive("CONTINUGO_CACHE_REPEATS", 3);
    let actions: Vec<_> = (0..info.moves)
        .map(|ply| crate::cells::actions(&cells(&replay, ply, info.resolution), info.resolution))
        .collect();
    // Warm code/allocator on early/middle/late, retaining no extractor cache.
    for ply in [0, info.moves / 2, info.moves - 1] {
        let sample = replay.position(ply).unwrap();
        let extractor = FeatureExtractor::new(&sample.game, 4).unwrap();
        black_box(extractor.moves_footprint(&actions[ply]).unwrap());
    }
    let mut baseline = Vec::new();
    let mut per_ply = vec![Vec::new(); info.moves];
    for _ in 0..repeats {
        let (mut coarse, mut fine) = (0.0, 0.0);
        for ply in 0..info.moves {
            let sample = replay.position(ply).unwrap();
            let tick = Instant::now();
            let extractor = FeatureExtractor::new(&sample.game, 4).unwrap();
            let c = tick.elapsed().as_secs_f64();
            black_box(&extractor.coarse);
            let tick = Instant::now();
            let move_extractor = FeatureExtractor::for_moves(&sample.game, SCHEMA_VERSION).unwrap();
            black_box(move_extractor.moves_footprint(&actions[ply]).unwrap());
            let f = tick.elapsed().as_secs_f64();
            coarse += c;
            fine += f;
            per_ply[ply].push(json!({"coarse_seconds":c,"fine_seconds":f}));
        }
        baseline.push(json!({"coarse_seconds":coarse,"fine_seconds":fine}));
    }
    let mut positions = Vec::new();
    let mut previous: Option<Snapshot> = None;
    let mut local_cache = LocalCache::new();
    for ply in 0..info.moves {
        let sample = replay.position(ply).unwrap();
        let tick = Instant::now();
        let (extractor, coarse_profile) = FeatureExtractor::profiled(&sample.game, 4).unwrap();
        let coarse_seconds = tick.elapsed().as_secs_f64();
        // Real inference drops the coarse extractor and prepares again for moves.
        let tick = Instant::now();
        let move_extractor = FeatureExtractor::for_moves(&sample.game, SCHEMA_VERSION).unwrap();
        let move_preparation_seconds = tick.elapsed().as_secs_f64();
        let reusable = previous
            .as_ref()
            .map(|parent| reusable_local_mask(parent, &sample.game, &actions[ply]))
            .unwrap_or_else(|| vec![false; actions[ply].len()]);
        if let Some(parent) = &previous {
            local_cache.invalidate(
                &changed_stones(&parent.game, &sample.game),
                sample.game.board().diameter(),
            );
        }
        let inherited: Vec<_> = actions[ply]
            .iter()
            .map(|a| local_cache.get(a.index()))
            .collect();
        let tick = Instant::now();
        let (moves, fine) = fine_profile(&move_extractor, &actions[ply], &reusable, &inherited);
        let fine_seconds = tick.elapsed().as_secs_f64();
        let ordinary = FeatureExtractor::new(&sample.game, 4).unwrap();
        bits_equal(&ordinary.coarse, &extractor.coarse);
        let expected = ordinary.moves_footprint(&actions[ply]).unwrap();
        bits_equal(&expected.exact, &moves.exact);
        assert_eq!(expected.legal, moves.legal);
        for (i, action) in actions[ply].iter().enumerate() {
            if moves.legal[i] != 0 {
                local_cache.insert(action.index(), moves.exact[i * 11 + 6]);
            }
        }
        let group_snapshots = groups(&extractor);
        let snapshot = Snapshot {
            game: sample.game.clone(),
            coarse: extractor.coarse,
            groups: group_snapshots,
            actions: actions[ply].clone(),
            moves,
        };
        let edge = previous.as_ref().map(|parent| audit(parent, &snapshot));
        if let Some(edge) = &edge {
            assert_eq!(
                edge.reusable_local_footprints as u64,
                fine.parent_reusable_footprint.calls
            );
        }
        positions.push(
            json!({"ply":ply,"stones":snapshot.game.position().stones().len(),
            "cells":cells(&replay,ply,info.resolution),"coordinates":snapshot.actions.len(),
            "legal":snapshot.moves.legal.iter().filter(|&&x| x!=0).count(),
            "baseline":per_ply[ply],"profiled_coarse_seconds":coarse_seconds,
            "coarse":coarse_profile,"move_preparation_seconds":move_preparation_seconds,
            "profiled_fine_seconds":fine_seconds,"fine":fine,"edge_from_parent":edge,
            "ancestor_cache_entries":local_cache.len()}),
        );
        previous = Some(snapshot);
    }
    let report = json!({"source":path,"sgf_sha256":format!("{:x}",Sha256::digest(text.as_bytes())),
        "info":info,"schema":SCHEMA_VERSION,"density":4,"repeats":repeats,"provenance":provenance,
        "workload":"positive-visit target cells at every pre-move ply; serial; no network/search/IPC",
        "allocator":"release libtest system allocator, not production mimalloc",
        "ancestor_cache_slots":LOCAL_SLOTS,"ancestor_cache_bytes":LOCAL_SLOTS*std::mem::size_of::<Option<(u32,f32)>>(),
        "baseline":baseline,"positions":positions});
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap();
    use std::io::Write;
    file.write_all(&serde_json::to_vec_pretty(&report).unwrap())
        .unwrap();
}

#[test]
fn simulated_local_cache_is_bounded_and_collision_checked() {
    let mut cache = LocalCache::new();
    let a = 500_500;
    let b = (0..COORDINATE_ACTIONS)
        .find(|&b| b != a && LocalCache::slot(a) == LocalCache::slot(b))
        .unwrap();
    cache.insert(a, 1.0);
    assert_eq!(cache.get(a), Some(1.0));
    assert_eq!(cache.get(b), None);
    cache.insert(b, 2.0);
    assert_eq!(cache.get(a), None);
    assert_eq!(cache.get(b), Some(2.0));
    cache.invalidate(
        &[Stone {
            color: Color::Black,
            spot: Action::from_index(b).unwrap().spot().unwrap(),
        }],
        110,
    );
    assert_eq!(cache.get(b), None);
    for id in 0..COORDINATE_ACTIONS {
        cache.insert(id, 0.0);
    }
    assert_eq!(cache.len(), LOCAL_SLOTS);
}

#[test]
fn diagnostic_stage_paths_preserve_feature_bits() {
    use continugo_rules::Board;
    let mut game = Game::new(Board::new(9).unwrap());
    let actions: Vec<_> = (0..COORDINATE_ACTIONS)
        .step_by(997)
        .map(|i| Action::from_index(i).unwrap())
        .collect();
    for i in 0..3 {
        let (profiled, stages) = FeatureExtractor::profiled(&game, 4).unwrap();
        let ordinary = FeatureExtractor::new(&game, 4).unwrap();
        bits_equal(&profiled.coarse, &ordinary.coarse);
        let mut merged = CoarseProfile::default();
        merged.merge(&stages);
        assert_eq!(
            serde_json::to_value(merged).unwrap(),
            serde_json::to_value(stages).unwrap()
        );
        let moves = FeatureExtractor::for_moves(&game, SCHEMA_VERSION).unwrap();
        let (profiled, _) = fine_profile(
            &moves,
            &actions,
            &vec![false; actions.len()],
            &vec![None; actions.len()],
        );
        let expected = ordinary.moves_footprint(&actions).unwrap();
        bits_equal(&profiled.exact, &expected.exact);
        assert_eq!(profiled.legal, expected.legal);
        game = game
            .play(Spot {
                x: 55 + 300 * i,
                y: 55,
            })
            .unwrap();
    }
}

#[test]
fn local_invalidation_covers_pass_capture_and_group_merge() {
    use continugo_rules::Board;
    let board = Board::new(9).unwrap();
    let r = board.stone_radius();
    let parent = Game::new(board).play(Spot { x: r, y: r }).unwrap();
    let prepared = parent.position().prepare();
    let capture = prepared
        .analyze_group(prepared.groups(Color::Black).next().unwrap())
        .capture_in_one
        .unwrap();
    let child = parent.play(capture).unwrap();
    assert_eq!(changed_stones(&parent, &child).len(), 2);
    let pass = parent.pass().unwrap();
    assert!(changed_stones(&parent, &pass).is_empty());
    for next in [&child, &pass] {
        let changes = changed_stones(&parent, next);
        let a = parent.position().prepare();
        let b = next.position().prepare();
        for y in (r..=1000 - r).step_by(13) {
            for x in (r..=1000 - r).step_by(13) {
                let spot = Spot { x, y };
                if safe(spot, &changes, board.diameter()) {
                    assert_eq!(a.local_liberty_spans(spot), b.local_liberty_spans(spot));
                }
            }
        }
    }
    let parent = Game::new(board)
        .play(Spot { x: 300, y: 500 })
        .unwrap()
        .pass()
        .unwrap()
        .play(Spot { x: 550, y: 500 })
        .unwrap()
        .pass()
        .unwrap();
    let child = parent.play(Spot { x: 425, y: 500 }).unwrap();
    assert_eq!(parent.position().groups(Color::Black).len(), 2);
    assert_eq!(child.position().groups(Color::Black).len(), 1);
    assert_eq!(changed_stones(&parent, &child).len(), 1);
}
