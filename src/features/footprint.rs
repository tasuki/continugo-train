//! New stone's local liberty footprint, with existing stones fixed as obstacles.
//! Values saturate at three discs. Free disjoint probes can certify the cap;
//! otherwise exact center spans feed a small raster that stops at the cap.
//! Small footprints retain their original resolution. No child/group analysis.

use continugo_rules::{LibertySpan, PreparedPosition, Spot};
use std::sync::{Mutex, OnceLock};

const CACHE_SLOTS: usize = 1024;

struct Entry {
    centers: [u128; SIDE],
    value: f32,
}

/// Bounded direct-mapped memoization of complete local center rasters. Hash
/// collisions only evict entries: full bitmap equality is required for a hit.
/// Each extractor owns at most ~2 MiB, shared safely by its move workers.
#[derive(Default)]
pub(super) struct Cache(Mutex<Vec<Option<Box<Entry>>>>);

impl Cache {
    fn get(&self, key: usize, centers: &[u128; SIDE]) -> Option<f32> {
        let slots = self.0.lock().unwrap();
        slots
            .get(key)?
            .as_ref()
            .filter(|e| e.centers == *centers)
            .map(|e| e.value)
    }

    fn insert(&self, key: usize, centers: [u128; SIDE], value: f32) {
        let mut slots = self.0.lock().unwrap();
        if slots.is_empty() {
            slots.resize_with(CACHE_SLOTS, || None);
        }
        slots[key] = Some(Box::new(Entry { centers, value }));
    }
}

fn cache_key(centers: &[u128; SIDE], first: usize, last: usize) -> usize {
    let mut hash = first as u64 ^ (last as u64).rotate_left(32);
    for &row in &centers[first..=last] {
        hash = (hash.rotate_left(9) ^ row as u64).wrapping_mul(0x9e3779b185ebca87);
        hash ^= ((row >> 64) as u64).wrapping_mul(0xc2b2ae3d27d4eb4f);
    }
    hash as usize & (CACHE_SLOTS - 1)
}

pub(super) const CAP: f32 = 3.0;
pub(super) const SAMPLES_PER_RADIUS: usize = 16;
const SIDE: usize = 128;
const ORIGIN: i32 = 64;

const fn circle_rows() -> [usize; SAMPLES_PER_RADIUS + 1] {
    let mut rows = [0; SAMPLES_PER_RADIUS + 1];
    let mut y = 0;
    while y <= SAMPLES_PER_RADIUS {
        rows[y] = (SAMPLES_PER_RADIUS * SAMPLES_PER_RADIUS - y * y).isqrt();
        y += 1;
    }
    rows
}

const CIRCLE: [usize; SAMPLES_PER_RADIUS + 1] = circle_rows();
const DISC_PIXELS: usize = {
    let mut area = 2 * CIRCLE[0] + 1;
    let mut y = 1;
    while y <= SAMPLES_PER_RADIUS {
        area += 2 * (2 * CIRCLE[y] + 1);
        y += 1;
    }
    area
};

const CAP_PIXELS: u32 = 3 * DISC_PIXELS as u32;

struct Probes {
    offsets: [Spot; 8],
    valid: u8,
    disjoint: [u8; 8],
}

impl Probes {
    fn new(radius: i32) -> Self {
        let d = 2 * radius;
        let a = (2 * d * d - 1).isqrt();
        let b = d - 1;
        // Four outer cardinal neighbors first. Diagonals also certify open
        // board corners, where only two cardinal neighbors can be in bounds.
        let offsets = [
            (a, 0),
            (-a, 0),
            (0, a),
            (0, -a),
            (b, b),
            (-b, b),
            (b, -b),
            (-b, -b),
        ]
        .map(|(x, y)| Spot { x, y });
        let mut valid = 0;
        let mut disjoint = [0; 8];
        for (i, p) in offsets.iter().enumerate() {
            let distance = p.x * p.x + p.y * p.y;
            if distance < d * d || distance >= 2 * d * d {
                continue;
            }
            valid |= 1 << i;
            for (j, q) in offsets.iter().enumerate() {
                let dx = bin(p.x, radius) as i32 - bin(q.x, radius) as i32;
                let dy = bin(p.y, radius) as i32 - bin(q.y, radius) as i32;
                if dx * dx + dy * dy > 4 * (SAMPLES_PER_RADIUS as i32).pow(2) {
                    disjoint[i] |= 1 << j;
                }
            }
        }
        Self {
            offsets,
            valid,
            disjoint,
        }
    }
}

fn probe_saturated(prepared: &PreparedPosition<'_>, spot: Spot, radius: i32) -> bool {
    static PROBES: [OnceLock<Probes>; 501] = [const { OnceLock::new() }; 501];
    let probes = PROBES[radius as usize].get_or_init(|| Probes::new(radius));
    let mut free = 0u8;
    for (i, offset) in probes.offsets.iter().enumerate() {
        if probes.valid & (1 << i) == 0 {
            continue;
        }
        let center = Spot {
            x: spot.x + offset.x,
            y: spot.y + offset.y,
        };
        if !prepared.is_local_liberty_center(spot, center) {
            continue;
        }
        let mut mates = free & probes.disjoint[i];
        while mates != 0 {
            let j = mates.trailing_zeros() as usize;
            if mates & probes.disjoint[j] != 0 {
                return true;
            }
            mates &= mates - 1;
        }
        free |= 1 << i;
    }
    false
}

fn bin(delta: i32, radius: i32) -> usize {
    // Nearest sample, ties toward positive coordinates. The annulus reaches
    // <sqrt(8)*radius; its footprint therefore fits in +/-62 samples at density 16.
    (ORIGIN + (2 * delta * SAMPLES_PER_RADIUS as i32 + radius).div_euclid(2 * radius)) as usize
}

fn spread(mask: u128, width: usize) -> u128 {
    if width == 0 {
        return mask;
    }
    let mut low = mask as u64;
    let mut high = (mask >> 64) as u64;
    low = (low >> width) | (high << (64 - width));
    high >>= width;
    let mut remaining = 2 * width;
    let mut step = 1;
    while remaining > 0 {
        let shift = step.min(remaining);
        high |= (high << shift) | (low >> (64 - shift));
        low |= low << shift;
        remaining -= shift;
        step *= 2;
    }
    (u128::from(high) << 64) | u128::from(low)
}

fn project_span(centers: &mut [u128; SIDE], span: LibertySpan, spot: Spot, radius: i32) -> usize {
    let y = bin(span.y - spot.y, radius);
    let left = bin(span.left - spot.x, radius);
    let right = bin(span.right - spot.x, radius);
    let mask = (u128::MAX >> (127 - (right - left))) << left;
    centers[y] |= mask;
    y
}

pub(super) fn area(prepared: &PreparedPosition<'_>, spot: Spot, radius: i32, cache: &Cache) -> f32 {
    if probe_saturated(prepared, spot, radius) {
        return CAP;
    }
    let mut centers = [0u128; SIDE];
    let mut first = SIDE;
    let mut last = 0;
    prepared.visit_local_liberty_spans(spot, |span| {
        let y = project_span(&mut centers, span, spot, radius);
        first = first.min(y);
        last = last.max(y);
    });
    if first == SIDE {
        return 0.0;
    }
    let key = cache_key(&centers, first, last);
    if let Some(value) = cache.get(key, &centers) {
        return value;
    }
    let value = capped_raster_area(&centers, first, last);
    cache.insert(key, centers, value);
    value
}

fn capped_raster_area(centers: &[u128; SIDE], first: usize, last: usize) -> f32 {
    let mut footprint = [0u128; SIDE];
    let pixels = dilate_impl::<true>(centers, first, last, &mut footprint);
    pixels as f32 / DISC_PIXELS as f32
}

#[cfg(test)]
fn raster_area(centers: &[u128; SIDE], first: usize, last: usize) -> f32 {
    let mut footprint = [0u128; SIDE];
    dilate(centers, first, last, &mut footprint);
    let pixels: u32 = footprint.iter().map(|row| row.count_ones()).sum();
    pixels as f32 / DISC_PIXELS as f32
}

#[cfg(test)]
fn dilate(centers: &[u128; SIDE], first: usize, last: usize, footprint: &mut [u128; SIDE]) {
    dilate_impl::<false>(centers, first, last, footprint);
}

fn dilate_impl<const EARLY_EXIT: bool>(
    centers: &[u128; SIDE],
    first: usize,
    last: usize,
    footprint: &mut [u128; SIDE],
) -> u32 {
    let mut pixels = 0;
    for (y, &mask) in centers.iter().enumerate().take(last + 1).skip(first) {
        if mask == 0 {
            continue;
        }
        let mut dy = 0;
        while dy <= SAMPLES_PER_RADIUS {
            let width = CIRCLE[dy];
            let covered = spread(mask, width);
            loop {
                if EARLY_EXIT {
                    pixels += (covered & !footprint[y + dy]).count_ones();
                }
                footprint[y + dy] |= covered;
                if dy != 0 {
                    if EARLY_EXIT {
                        pixels += (covered & !footprint[y - dy]).count_ones();
                    }
                    footprint[y - dy] |= covered;
                }
                if EARLY_EXIT && pixels >= CAP_PIXELS {
                    return CAP_PIXELS;
                }
                dy += 1;
                if dy > SAMPLES_PER_RADIUS || CIRCLE[dy] != width {
                    break;
                }
            }
        }
    }
    pixels
}

#[cfg(test)]
mod coverage_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::FeatureExtractor;
    use continugo_rules::Stone;
    use std::{collections::HashMap, env, fs, hint::black_box, time::Instant};

    #[test]
    fn bitset_dilation_preserves_unions_and_single_disc_units() {
        for width in 0..=SAMPLES_PER_RADIUS {
            let mask = ((1u128 << 80) - 1) << 24;
            let mut expected = mask;
            for dx in 1..=width {
                expected |= (mask << dx) | (mask >> dx);
            }
            assert_eq!(spread(mask, width), expected);
        }
        let mut seed = 0x123456789abcdefu128;
        for _ in 0..16 {
            seed = seed.wrapping_mul(0x5851f42d4c957f2d).wrapping_add(1);
            let mask = seed & (((1u128 << 96) - 1) << 16);
            for width in 0..=SAMPLES_PER_RADIUS {
                let mut expected = mask;
                for dx in 1..=width {
                    expected |= (mask << dx) | (mask >> dx);
                }
                assert_eq!(spread(mask, width), expected);
            }
        }
        let mut centers = [0; SIDE];
        centers[64] = 1u128 << 64;
        assert_eq!(raster_area(&centers, 64, 64), 1.0);
        centers[64] = (1u128 << 40) | (1u128 << 88);
        assert_eq!(raster_area(&centers, 64, 64), 2.0);
        centers[64] = (1u128 << 64) | (1u128 << 65);
        let overlapping = raster_area(&centers, 64, 64);
        assert!(overlapping > 1.0 && overlapping < 2.0);
    }

    #[test]
    fn capped_dilation_matches_the_complete_union_below_and_above_three() {
        let mut seed = 0x123456789abcdefu128;
        for trial in 0..128 {
            let mut centers = [0u128; SIDE];
            for _ in 0..trial % 12 {
                seed = seed.wrapping_mul(0x5851f42d4c957f2d).wrapping_add(1);
                let x = 20 + (seed as usize) % 88;
                let y = 20 + ((seed >> 64) as usize) % 88;
                centers[y] |= 1u128 << x;
            }
            assert_eq!(
                capped_raster_area(&centers, 16, 111),
                raster_area(&centers, 16, 111).min(CAP)
            );
        }
        // Three almost-touching discs overlap by two pixels: do not round this
        // sub-cap result up to three merely because there are three centers.
        let mut centers = [0; SIDE];
        centers[64] = (1u128 << 32) | (1u128 << 64) | (1u128 << 96);
        let complete = raster_area(&centers, 64, 64);
        assert!(complete < CAP && complete > 2.99);
        assert_eq!(capped_raster_area(&centers, 64, 64), complete);
    }

    #[test]
    fn probe_certificates_are_disjoint_and_do_not_replace_small_pockets() {
        use continugo_rules::{Board, Position};
        for radius in 1..=500 {
            let probes = Probes::new(radius);
            for (i, p) in probes.offsets.iter().enumerate() {
                if probes.valid & (1 << i) == 0 {
                    continue;
                }
                let d = 2 * radius;
                let distance = p.x * p.x + p.y * p.y;
                assert!(distance >= d * d && distance < 2 * d * d);
                for (j, q) in probes.offsets.iter().enumerate() {
                    if probes.disjoint[i] & (1 << j) != 0 {
                        let dx = bin(p.x, radius) as i32 - bin(q.x, radius) as i32;
                        let dy = bin(p.y, radius) as i32 - bin(q.y, radius) as i32;
                        assert!(dx * dx + dy * dy > 4 * (SAMPLES_PER_RADIUS as i32).pow(2));
                    }
                }
            }
        }
        for size in [3, 9, 13, 19] {
            let board = Board::new(size).unwrap();
            let r = board.stone_radius();
            let position = Position::new(board);
            for p in [
                Spot { x: 500, y: 500 },
                Spot { x: r, y: r },
                Spot {
                    x: 1000 - r,
                    y: 1000 - r,
                },
            ] {
                assert!(probe_saturated(&position.prepare(), p, r));
                assert_eq!(area(&position.prepare(), p, r, &Cache::default()), CAP);
            }
        }
    }

    #[test]
    fn cache_is_bounded_and_collisions_never_reuse_a_different_geometry() {
        let cache = Cache::default();
        let mut a = [0; SIDE];
        a[64] = 1 << 64;
        let mut b = a;
        b[64] |= 1 << 80;
        cache.insert(0, a, 1.0);
        assert_eq!(cache.get(0, &a), Some(1.0));
        assert_eq!(cache.get(0, &b), None);
        cache.insert(0, b, 2.0);
        assert_eq!(cache.get(0, &a), None);
        assert_eq!(cache.get(0, &b), Some(2.0));
        for i in 0..2 * CACHE_SLOTS {
            cache.insert(i % CACHE_SLOTS, a, i as f32);
        }
        assert_eq!(cache.0.lock().unwrap().len(), CACHE_SLOTS);
    }

    #[test]
    fn raster_tracks_continuous_local_area_across_stone_sizes() {
        use continugo_rules::{Board, Color, Position};
        for size in [3, 9, 13, 19, 500] {
            let board = Board::new(size).unwrap();
            let r = board.stone_radius();
            let mut position = Position::new(board);
            for spot in [
                Spot { x: r, y: r },
                Spot { x: 500, y: 500 },
                Spot { x: 800, y: 200 },
            ] {
                if let Some(next) = position.play(Stone {
                    color: Color::Black,
                    spot,
                }) {
                    position = next;
                }
            }
            let prepared = position.prepare();
            let cache = Cache::default();
            for spot in [
                Spot { x: r, y: r },
                Spot { x: 500, y: 500 },
                Spot { x: 750, y: 750 },
                Spot {
                    x: 500 + board.diameter(),
                    y: 500,
                },
            ] {
                if !board.contains(spot) {
                    continue;
                }
                let reference =
                    crate::liberty_area::in_stone_discs(&prepared.local_liberty_spans(spot), r)
                        .min(f64::from(CAP));
                let actual = f64::from(area(&prepared, spot, r, &cache));
                // This is an area-feature approximation, not a rule predicate.
                // Quantization is explicit; isolated centers have exact unit area.
                assert!(
                    (actual - reference).abs() <= 0.06 * reference.max(1.0),
                    "size={size},p={spot:?}: {actual} vs {reference}"
                );
                assert_eq!(actual as f32, area(&prepared, spot, r, &cache));
            }
        }
    }

    #[test]
    fn obstacles_are_color_blind_and_captured_stones_are_not_removed() {
        use continugo_rules::{Board, Color, Position};
        let board = Board::new(9).unwrap();
        let r = board.stone_radius();
        let corner = Spot { x: r, y: r };
        let black = Position::new(board)
            .play(Stone {
                color: Color::Black,
                spot: corner,
            })
            .unwrap();
        let white = Position::new(board)
            .play(Stone {
                color: Color::White,
                spot: corner,
            })
            .unwrap();
        let prepared = black.prepare();
        let capture = prepared
            .analyze_group(&black.groups(Color::Black)[0])
            .capture_in_one
            .unwrap();
        let cache = Cache::default();
        let fixed = area(&prepared, capture, r, &cache);
        assert_eq!(fixed, area(&white.prepare(), capture, r, &cache));
        let child = prepared
            .analyze_move(
                Stone {
                    color: Color::White,
                    spot: capture,
                },
                0,
            )
            .unwrap();
        assert_eq!(child.captured_stones, 1);
        // Saturation may hide the scalar difference, but the old stone is
        // still a blocker in the current-position geometry, never removed.
        assert!(!prepared.is_local_liberty_center(capture, corner));
        assert!(
            child
                .position
                .prepare()
                .is_local_liberty_center(capture, corner)
        );
        assert!(area(&child.position.prepare(), capture, r, &cache) <= CAP);
    }

    #[test]
    #[ignore = "local accuracy/timing audit; set CONTINUGO_FOOTPRINT_CASES and CONTINUGO_BENCH_SGF"]
    fn profile_local_footprints() {
        let cases: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(env::var("CONTINUGO_FOOTPRINT_CASES").unwrap()).unwrap(),
        )
        .unwrap();
        let sgf = fs::read_to_string(env::var("CONTINUGO_BENCH_SGF").unwrap()).unwrap();
        let mut reports = Vec::new();
        for case in cases.as_array().unwrap() {
            let ply = case["ply"].as_u64().unwrap() as usize;
            let sample = crate::replay::load_position(&sgf, ply).unwrap();
            let extractor = FeatureExtractor::new(&sample.game, 4).unwrap();
            let cells: Vec<_> = case["cells"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            let actions = crate::cells::actions(&cells, extractor.resolution);
            let r = sample.game.board().stone_radius();
            let cache = Cache::default();
            let mut errors = Vec::new();
            let mut relative_max: f64 = 0.0;
            let mut continuous_seconds = 0.0;
            let mut raster_seconds = 0.0;
            let mut spans_seconds = 0.0;
            let mut worst = serde_json::Value::Null;
            let mut maximum: f64 = 0.0;
            for action in actions.iter().step_by(8) {
                let spot = action.spot().unwrap();
                if extractor
                    .prepared
                    .analyze_move(
                        Stone {
                            color: sample.game.on_move(),
                            spot,
                        },
                        0,
                    )
                    .is_none()
                {
                    continue;
                }
                let tick = Instant::now();
                let spans = extractor.prepared.local_liberty_spans(spot);
                spans_seconds += tick.elapsed().as_secs_f64();
                let tick = Instant::now();
                let reference = crate::liberty_area::in_stone_discs(&spans, r).min(f64::from(CAP));
                continuous_seconds += tick.elapsed().as_secs_f64();
                let tick = Instant::now();
                let actual = f64::from(area(&extractor.prepared, spot, r, &cache));
                raster_seconds += tick.elapsed().as_secs_f64();
                let difference = (actual - reference).abs();
                if difference > maximum {
                    maximum = difference;
                    worst = serde_json::json!({"action":action.index(),"reference":reference,"raster":actual});
                }
                relative_max = relative_max.max(difference / reference.max(1.0));
                errors.push(difference);
            }
            errors.sort_by(f64::total_cmp);
            let n = errors.len();
            let report = serde_json::json!({"ply":ply,"legal_sampled":n,"density":SAMPLES_PER_RADIUS,"cap":CAP,
                "spans_seconds":spans_seconds,"continuous_area_seconds":continuous_seconds,"local_raster_seconds":raster_seconds,
                "mean_abs_discs":errors.iter().sum::<f64>()/n.max(1) as f64,
                "p95_abs_discs":errors.get(n*95/100),"max_abs_discs":maximum,
                "max_relative_to_at_least_one_disc":relative_max,"worst":worst});
            println!("{report}");
            reports.push(report);
        }
        if let Ok(path) = env::var("CONTINUGO_FOOTPRINT_PROFILE_OUTPUT") {
            fs::write(path, serde_json::to_string_pretty(&reports).unwrap()).unwrap();
        }
    }

    #[test]
    #[ignore = "explicit CPU diagnostic; set CONTINUGO_FOOTPRINT_CASES and CONTINUGO_BENCH_SGF"]
    fn profile_candidate_stages() {
        let cases: serde_json::Value = serde_json::from_str(
            &fs::read_to_string(env::var("CONTINUGO_FOOTPRINT_CASES").unwrap()).unwrap(),
        )
        .unwrap();
        let sgf = fs::read_to_string(env::var("CONTINUGO_BENCH_SGF").unwrap()).unwrap();
        let mut reports = Vec::new();
        let keys_only = env::var_os("CONTINUGO_FOOTPRINT_KEYS_ONLY").is_some();
        let stride = if keys_only { 1 } else { 8 };
        for case in cases.as_array().unwrap() {
            let ply = case["ply"].as_u64().unwrap() as usize;
            let sample = crate::replay::load_position(&sgf, ply).unwrap();
            let extractor = FeatureExtractor::new(&sample.game, 4).unwrap();
            let cells: Vec<_> = case["cells"]
                .as_array()
                .unwrap()
                .iter()
                .map(|c| c.as_u64().unwrap() as usize)
                .collect();
            let actions = crate::cells::actions(&cells, extractor.resolution);
            let mut stages = [0.0; 6];
            let mut cache = HashMap::new();
            let mut hits = 0;
            let mut legal = 0;
            let mut total_spans = 0;
            let mut count = 0;
            for action in actions.iter().step_by(stride) {
                count += 1;
                let spot = action.spot().unwrap();
                let tick = Instant::now();
                let analysis = extractor.prepared.analyze_move(
                    Stone {
                        color: sample.game.on_move(),
                        spot,
                    },
                    0,
                );
                stages[0] += tick.elapsed().as_secs_f64();
                let Some(analysis) = analysis else { continue };
                legal += 1;
                let tick = Instant::now();
                let group = analysis
                    .position
                    .groups(sample.game.on_move())
                    .into_iter()
                    .find(|members| {
                        members
                            .iter()
                            .any(|&i| analysis.position.stones()[i].spot == spot)
                    })
                    .unwrap();
                stages[1] += tick.elapsed().as_secs_f64();
                let tick = Instant::now();
                let spans = analysis.position.liberty_spans(&group);
                stages[2] += tick.elapsed().as_secs_f64();
                total_spans += spans.len();
                let tick = Instant::now();
                let original = if keys_only {
                    0.0
                } else {
                    black_box(crate::liberty_area::in_stone_discs(
                        &spans,
                        sample.game.board().stone_radius(),
                    ))
                };
                stages[3] += tick.elapsed().as_secs_f64();
                let tick = Instant::now();
                let optimized = if keys_only {
                    0.0
                } else {
                    black_box(crate::liberty_area::candidate_in_stone_discs(
                        &spans,
                        sample.game.board().stone_radius(),
                    ))
                };
                stages[4] += tick.elapsed().as_secs_f64();
                assert_eq!(
                    original.to_bits(),
                    optimized.to_bits(),
                    "ply={ply}, action={action:?}"
                );
                let tick = Instant::now();
                let origin = spans.first().unwrap();
                let key: Vec<_> = spans
                    .iter()
                    .map(|s| [s.y - origin.y, s.left - origin.left, s.right - origin.left])
                    .collect();
                let cached = if let Some(&area) = cache.get(&key) {
                    hits += 1;
                    area
                } else {
                    let area = if keys_only {
                        0.0
                    } else {
                        crate::liberty_area::candidate_in_stone_discs(
                            &spans,
                            sample.game.board().stone_radius(),
                        )
                    };
                    cache.insert(key, area);
                    area
                };
                stages[5] += tick.elapsed().as_secs_f64();
                assert!((optimized - cached).abs() < 1e-8);
            }
            let report = serde_json::json!({"ply":ply,"candidates":count,"legal":legal,
                "keys_only":keys_only,"stride":stride,
                "total_spans":total_spans,"validation_seconds":stages[0],
                "groups_seconds":stages[1],"spans_seconds":stages[2],"area_seconds":stages[3],
                "optimized_area_seconds":stages[4],"cached_area_seconds":stages[5],
                "cache_entries":cache.len(),"cache_hits":hits});
            println!("{report}");
            reports.push(report);
        }
        if let Ok(path) = env::var("CONTINUGO_FOOTPRINT_PROFILE_OUTPUT") {
            fs::write(path, serde_json::to_string_pretty(&reports).unwrap()).unwrap();
        }
    }
}
