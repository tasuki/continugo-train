//! Coverage, rather than average-area accuracy. Optional JSON comes from the
//! same passing fixtures and actual production projection/dilation functions.

use super::*;
use continugo_rules::{Board, Color, Position, Stone};
use std::{env, fs};

struct Fixture {
    id: &'static str,
    title: &'static str,
    note: &'static str,
    radius: i32,
    candidate: Spot,
    spans: Vec<LibertySpan>,
    position: Option<Position>,
    expected_area: Option<f32>,
    highlight: Option<Spot>,
    zoom: Option<[f64; 4]>,
}

fn span(y: i32, left: i32, right: i32) -> LibertySpan {
    LibertySpan { y, left, right }
}

fn position(board: Board, spots: &[(i32, i32)]) -> Position {
    let mut result = Position::new(board);
    for &(x, y) in spots {
        result = result
            .play(Stone {
                color: Color::Black,
                spot: Spot { x, y },
            })
            .unwrap();
    }
    result
}

fn real(
    id: &'static str,
    title: &'static str,
    note: &'static str,
    position: Position,
    candidate: Spot,
    expected_area: Option<f32>,
    highlight: Option<Spot>,
) -> Fixture {
    let spans = position.prepare().local_liberty_spans(candidate);
    Fixture {
        id,
        title,
        note,
        radius: position.board().stone_radius(),
        candidate,
        spans,
        position: Some(position),
        expected_area,
        highlight,
        zoom: None,
    }
}

fn fixtures() -> Vec<Fixture> {
    let p = Spot { x: 500, y: 500 };
    let synthetic = |id, title, note, spans, expected_area, highlight, zoom| Fixture {
        id,
        title,
        note,
        radius: 55,
        candidate: p,
        spans,
        position: None,
        expected_area,
        highlight,
        zoom,
    };
    let mut result = vec![
        synthetic(
            "off-grid",
            "One off-grid center / word boundary",
            "Synthetic exact-span fixture: (501,610) is between raster samples and maps to bit column 64. It must still make one full disc.",
            vec![span(610, 501, 501)],
            Some(1.0),
            Some(Spot { x: 501, y: 610 }),
            None,
        ),
        real(
            "rotated-eye",
            "Real geometry: off-grid one-point eye",
            "Three fixed stones plus the candidate bound an isolated integer center at (500,500). Its relative offset (-66,-88) is NOT on the 16/r grid. Outer liberties are present too.",
            position(
                Board::new(9).unwrap(),
                &[(434, 412), (412, 566), (588, 434)],
            ),
            Spot { x: 566, y: 588 },
            None,
            Some(Spot { x: 500, y: 500 }),
        ),
        synthetic(
            "horizontal-slit",
            "One-integer-row slit",
            "Synthetic exact-span fixture: 41 integer centers on y=501. The slit is much thinner than one raster cell; none may be discarded.",
            vec![span(501, 610, 650)],
            None,
            None,
            None,
        ),
        synthetic(
            "vertical-slit",
            "One-integer-column slit",
            "Synthetic exact-span fixture: 47 integer centers at x=610, crossing many sample phases. The resulting capsule must remain visible.",
            (477..=523).map(|y| span(y, 610, 610)).collect(),
            None,
            None,
            None,
        ),
        synthetic(
            "two-pockets",
            "Two separate one-point pockets",
            "Synthetic exact-span fixture: the centers at (390,500) and (610,500) produce two disjoint raster discs, not a vanished small pocket.",
            vec![span(500, 390, 390), span(500, 610, 610)],
            Some(2.0),
            None,
            None,
        ),
        synthetic(
            "gap-limit",
            "Resolution limit: a 1-unit gap can close",
            "Both centers survive, but a real 1-unit gap between their discs becomes one touching raster pixel. This is NOT a topology/eye-count guarantee. Insets enlarge the gap.",
            vec![span(444, 610, 610), span(555, 610, 610)],
            None,
            None,
            Some([599.0, 494.0, 22.0, 11.0]),
        ),
    ];
    result[1].zoom = Some([438.0, 438.0, 124.0, 124.0]);
    result.push(real("corner-eye", "Real geometry: one touching corner center",
        "Seven existing stones and the candidate leave exactly (166,166). It is not found by lucky point sampling: the exact row span is projected. All four board reflections are tested.",
        position(Board::new(3).unwrap(), &[(830,166),(166,498),(498,498),(830,498),(166,830),(498,830),(830,830)]),
        Spot{x:498,y:166}, Some(1.0), Some(Spot{x:166,y:166})));
    result.push(real("correct-zero", "Correct zero, not a missing local pocket",
        "This move is legal through a group with a distant eye. Every LOCAL center is blocked, so zero is correct; the distant eye is deliberately outside this feature.",
        position(Board::new(3).unwrap(), &[(498,166),(830,166),(166,498),(498,498),(830,498),(166,830),(498,830)]),
        Spot{x:830,y:830}, Some(0.0), None));
    result.push(real("border-annulus", "Real geometry: board-edge clipping",
        "Empty size-9 board, candidate tangent to the left border. Exact border-clipped centers and their footprint must fit the 128x128 buffer without cropping.",
        Position::new(Board::new(9).unwrap()), Spot{x:55,y:500}, None, None));
    result
}

fn assert_exact_geometry(f: &Fixture) {
    let Some(position) = &f.position else { return };
    let board = position.board();
    let expected = |q| {
        board.contains(q)
            && board.adjacent(f.candidate, q)
            && !board.overlaps(f.candidate, q)
            && position.stones().iter().all(|s| !board.overlaps(s.spot, q))
    };
    let mut actual = vec![false; 1_000_000];
    for s in &f.spans {
        for x in s.left..=s.right {
            let q = Spot { x, y: s.y };
            assert!(expected(q), "{}: spurious center {q:?}", f.id);
            actual[(s.y * 1000 + x) as usize] = true;
        }
    }
    let reach = (2 * board.diameter() * board.diameter() - 1).isqrt();
    let low = board.stone_radius();
    let high = 1000 - low;
    for y in (f.candidate.y - reach).max(low)..=(f.candidate.y + reach).min(high) {
        for x in (f.candidate.x - reach).max(low)..=(f.candidate.x + reach).min(high) {
            assert_eq!(
                actual[(y * 1000 + x) as usize],
                expected(Spot { x, y }),
                "{}: missing/spurious ({x},{y})",
                f.id
            );
        }
    }
}

fn projected(f: &Fixture) -> ([u128; SIDE], usize, usize) {
    let mut centers = [0; SIDE];
    let mut first = SIDE;
    let mut last = 0;
    for &span in &f.spans {
        let y = project_span(&mut centers, span, f.candidate, f.radius);
        first = first.min(y);
        last = last.max(y);
    }
    (centers, first, last)
}

// Deliberately slow independent oracle: stamp each disc pixel individually,
// without the production spread(), word shifts, or circular row table.
fn naive_dilation(centers: &[u128; SIDE]) -> [u128; SIDE] {
    let mut result = [0; SIDE];
    let r = SAMPLES_PER_RADIUS as i32;
    for (y, &row) in centers.iter().enumerate() {
        let mut row = row;
        while row != 0 {
            let x = row.trailing_zeros() as i32;
            row &= row - 1;
            for dy in -r..=r {
                for dx in -r..=r {
                    if dx * dx + dy * dy <= r * r {
                        result[(y as i32 + dy) as usize] |= 1u128 << (x + dx);
                    }
                }
            }
        }
    }
    result
}

#[test]
fn every_radius_and_integer_phase_has_full_disc_margin() {
    for radius in 1i32..=500 {
        let reach = (8 * radius * radius - 1).isqrt();
        let mut previous = 0;
        for delta in -reach..=reach {
            let index = bin(delta, radius);
            assert!(
                index >= SAMPLES_PER_RADIUS && index + SAMPLES_PER_RADIUS < SIDE,
                "radius={radius}, delta={delta}, bin={index}"
            );
            // Independent integer error bound, not a second call to bin():
            // the represented coordinate is within half a pixel of the source.
            let error = (index as i32 - ORIGIN) * radius - delta * SAMPLES_PER_RADIUS as i32;
            assert!(2 * error.abs() <= radius);
            if 2 * error.abs() == radius {
                assert!(error >= 0);
            }
            assert!(index >= previous);
            previous = index;
        }
    }
    assert_eq!(bin(-1, 32), 64); // Exact half-cell tie rounds up.
    assert_eq!(bin(1, 32), 65);
}

#[test]
fn every_possible_single_center_survives_word_boundaries_and_buffer_edges() {
    for y in SAMPLES_PER_RADIUS..SIDE - SAMPLES_PER_RADIUS {
        for x in SAMPLES_PER_RADIUS..SIDE - SAMPLES_PER_RADIUS {
            let mut centers = [0; SIDE];
            centers[y] = 1u128 << x;
            let mut footprint = [0; SIDE];
            dilate(&centers, y, y, &mut footprint);
            assert_ne!(footprint[y] & (1u128 << x), 0, "({x},{y}) vanished");
            assert_eq!(
                footprint,
                naive_dilation(&centers),
                "({x},{y}) disc shape mismatch"
            );
            assert_eq!(
                footprint
                    .iter()
                    .map(|row| row.count_ones() as usize)
                    .sum::<usize>(),
                DISC_PIXELS,
                "({x},{y}) was clipped or shifted incorrectly"
            );
        }
    }
}

#[test]
fn one_touching_corner_center_survives_all_board_reflections() {
    let source = [
        (830, 166),
        (166, 498),
        (498, 498),
        (830, 498),
        (166, 830),
        (498, 830),
        (830, 830),
    ];
    for flip_x in [false, true] {
        for flip_y in [false, true] {
            let transform = |(x, y)| {
                (
                    if flip_x { 1000 - x } else { x },
                    if flip_y { 1000 - y } else { y },
                )
            };
            let spots: Vec<_> = source.into_iter().map(transform).collect();
            let position = position(Board::new(3).unwrap(), &spots);
            let (x, y) = transform((498, 166));
            let p = Spot { x, y };
            let (x, y) = transform((166, 166));
            let spans = position.prepare().local_liberty_spans(p);
            assert_eq!(spans, vec![span(y, x, x)]);
            assert_eq!(area(&position.prepare(), p, 166, &Cache::default()), 1.0);
        }
    }
}

#[test]
fn visual_fixtures_preserve_every_exact_center_and_match_independent_disc_stamps() {
    let mut records = Vec::new();
    for f in fixtures() {
        assert_exact_geometry(&f);
        let (centers, first, last) = projected(&f);
        let mut footprint = [0; SIDE];
        if first != SIDE {
            dilate(&centers, first, last, &mut footprint);
        }
        assert_eq!(
            footprint,
            naive_dilation(&centers),
            "{}: disc union mismatch",
            f.id
        );
        let pixels: u32 = footprint.iter().map(|row| row.count_ones()).sum();
        let value = pixels as f32 / DISC_PIXELS as f32;
        let mut exact_points = 0;
        for s in &f.spans {
            for x in s.left..=s.right {
                let bx = bin(x - f.candidate.x, f.radius);
                let by = bin(s.y - f.candidate.y, f.radius);
                assert_ne!(
                    centers[by] & (1u128 << bx),
                    0,
                    "{}: ({x},{}) lost in projection",
                    f.id,
                    s.y
                );
                assert_ne!(
                    footprint[by] & (1u128 << bx),
                    0,
                    "{}: ({x},{}) lost in dilation",
                    f.id,
                    s.y
                );
                exact_points += 1;
            }
        }
        assert_eq!(f.spans.is_empty(), pixels == 0, "{}: false zero", f.id);
        if let Some(expected) = f.expected_area {
            assert_eq!(value, expected, "{}", f.id);
        }
        if let Some(q) = f.highlight {
            assert!(
                f.spans
                    .iter()
                    .any(|s| s.y == q.y && (s.left..=s.right).contains(&q.x)),
                "{}: highlighted center missing",
                f.id
            );
        }
        if f.id == "rotated-eye" {
            let q = f.highlight.unwrap();
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if dx != 0 || dy != 0 {
                        assert!(
                            !f.spans.iter().any(
                                |s| s.y == q.y + dy && (s.left..=s.right).contains(&(q.x + dx))
                            ),
                            "eye was not isolated"
                        );
                    }
                }
            }
        }
        if f.id == "gap-limit" {
            assert!(value < 2.0 && value > 1.99);
        }
        let candidate_legal = f.position.as_ref().map(|p| {
            p.prepare()
                .play(Stone {
                    color: Color::Black,
                    spot: f.candidate,
                })
                .is_some()
        });
        if f.id == "correct-zero" {
            assert_eq!(candidate_legal, Some(true));
        }
        if let Some(position) = &f.position {
            let cache = Cache::default();
            for _ in 0..2 {
                assert_eq!(
                    area(&position.prepare(), f.candidate, f.radius, &cache),
                    value.min(CAP)
                );
            }
        }
        records.push(serde_json::json!({"id":f.id,"title":f.title,"note":f.note,
            "kind":if f.position.is_some(){"real fixed-obstacle geometry"}else{"synthetic exact spans: projection/dilation only"},
            "radius":f.radius,"candidate":[f.candidate.x,f.candidate.y],"candidate_legal":candidate_legal,
            "stones":f.position.as_ref().map(|p|p.stones().iter().map(|s|[s.spot.x,s.spot.y]).collect::<Vec<_>>()).unwrap_or_default(),
            "spans":f.spans.iter().map(|s|[s.y,s.left,s.right]).collect::<Vec<_>>(),
            "highlight":f.highlight.map(|q|[q.x,q.y]),"zoom":f.zoom,
            "centers":centers.iter().map(|row|format!("{row:032x}")).collect::<Vec<_>>(),
            "footprint":footprint.iter().map(|row|format!("{row:032x}")).collect::<Vec<_>>(),
            "exact_centers":exact_points,"occupied_center_bins":centers.iter().map(|row|row.count_ones()).sum::<u32>(),
            "projected_missing":0,"dilation_missing":0,"raster_area_discs":value,"feature_value_discs":value.min(CAP),
            "reference_area_discs":crate::liberty_area::in_stone_discs(&f.spans,f.radius)}));
    }
    if let Ok(path) = env::var("CONTINUGO_FOOTPRINT_COVERAGE_REPORT") {
        // Export cannot claim these checks passed if someone runs only this
        // fixture test rather than the complete coverage-test module.
        every_radius_and_integer_phase_has_full_disc_margin();
        every_possible_single_center_survives_word_boundaries_and_buffer_edges();
        one_touching_corner_center_survives_all_board_reflections();
        let axis_phases: usize = (1i32..=500)
            .map(|r| (2 * (8 * r * r - 1).isqrt() + 1) as usize)
            .sum();
        let report = serde_json::json!({"passed":true,"side":SIDE,"origin":ORIGIN,
            "samples_per_radius":SAMPLES_PER_RADIUS,"disc_pixels":DISC_PIXELS,"feature_cap":CAP,
            "axis_phases_checked":axis_phases,"single_center_positions_checked":(SIDE-2*SAMPLES_PER_RADIUS).pow(2),
            "corner_reflections_checked":4,"fixtures":records});
        fs::write(path, serde_json::to_string_pretty(&report).unwrap()).unwrap();
    }
}
