use super::*;
use crate::features::FeatureExtractor;
use continugo_rules::Stone;
use std::collections::BTreeSet;

fn points(spans: &[LibertySpan]) -> BTreeSet<(i32, i32)> {
    spans
        .iter()
        .flat_map(|s| (s.left..=s.right).map(move |x| (x, s.y)))
        .collect()
}

#[test]
fn intervals_and_extrema_match_point_geometry() {
    let a = [
        LibertySpan {
            y: 20,
            left: 4,
            right: 12,
        },
        LibertySpan {
            y: 21,
            left: 7,
            right: 7,
        },
        LibertySpan {
            y: 23,
            left: 0,
            right: 5,
        },
    ];
    let b = [
        LibertySpan {
            y: 20,
            left: 3,
            right: 5,
        },
        LibertySpan {
            y: 20,
            left: 9,
            right: 15,
        },
        LibertySpan {
            y: 21,
            left: 7,
            right: 8,
        },
    ];
    assert_eq!(
        points(&intersect(&a, &b)),
        points(&a).intersection(&points(&b)).copied().collect()
    );
    assert!(points(&intersect(&a, &b)).contains(&(7, 21)));
    for v in DIRECTIONS.into_iter().chain([(7, -3), (-11, 2)]) {
        let p = extreme(&a, v, Spot { x: 6, y: 21 }).unwrap();
        let value = |x: i32, y: i32| i64::from(v.0) * i64::from(x) + i64::from(v.1) * i64::from(y);
        assert_eq!(
            value(p.x, p.y),
            points(&a).iter().map(|&(x, y)| value(x, y)).max().unwrap()
        );
    }
    let sites = [
        Spot { x: 0, y: 10 },
        Spot { x: 15, y: 20 },
        Spot { x: 9, y: 30 },
    ];
    let expected = points(&a)
        .into_iter()
        .map(|(x, y)| Spot { x, y })
        .max_by_key(|&p| {
            (
                sites.iter().map(|&q| distance(p, q)).min().unwrap(),
                -(p.x + 1000 * p.y),
            )
        });
    assert_eq!(farthest(&a, &sites), expected);
    assert_eq!(
        points(&subtract(&a, &b)),
        points(&a).difference(&points(&b)).copied().collect()
    );
}

#[test]
fn span_components_preserve_islands_and_diagonal_single_points() {
    let spans = [
        LibertySpan {
            y: 20,
            left: 10,
            right: 10,
        },
        LibertySpan {
            y: 20,
            left: 50,
            right: 50,
        },
        LibertySpan {
            y: 21,
            left: 11,
            right: 11,
        },
        LibertySpan {
            y: 23,
            left: 10,
            right: 10,
        },
    ];
    let pieces = components(&spans);
    assert_eq!(pieces.len(), 3);
    assert!(
        pieces
            .iter()
            .any(|p| points(p) == BTreeSet::from([(10, 20), (11, 21)]))
    );
    assert_eq!(pieces.iter().map(|p| points(p).len()).sum::<usize>(), 4);
}

#[test]
fn join_sweep_preserves_a_single_coordinate_three_way_overlap() {
    let local = vec![
        vec![LibertySpan {
            y: 20,
            left: 10,
            right: 20,
        }],
        vec![LibertySpan {
            y: 20,
            left: 15,
            right: 25,
        }],
        vec![LibertySpan {
            y: 20,
            left: 18,
            right: 18,
        }],
    ];
    let regions = join_regions(&local, &[0, 1, 2]);
    assert_eq!(points(&regions[&vec![0, 1, 2]]), BTreeSet::from([(18, 20)]));
    for x in 0..30 {
        let p = Spot { x, y: 20 };
        let joined: Vec<_> = (0..3).filter(|&i| contains(&local[i], p)).collect();
        if joined.len() >= 2 {
            assert!(contains(&regions[&joined], p));
        } else {
            assert!(!regions.values().any(|spans| contains(spans, p)));
        }
    }
}

#[test]
fn real_three_group_join_inside_cell_with_all_stones_outside() {
    let game = Game::new(Board::new(9).unwrap())
        .play(Spot { x: 390, y: 500 })
        .unwrap()
        .play(Spot { x: 800, y: 800 })
        .unwrap()
        .play(Spot { x: 610, y: 500 })
        .unwrap()
        .pass()
        .unwrap()
        .play(Spot { x: 500, y: 690 })
        .unwrap()
        .pass()
        .unwrap();
    let (input, _) = FeatureExtractor::with_geometry(&game, 4, 5, false).unwrap();
    let geometry = input.geometry.unwrap();
    let cell = cells::cell(560500, input.resolution);
    let list = generate(&game, &geometry, input.resolution, &[cell], 8, 0).unwrap();
    let rect = cell_rect(cell, input.resolution);
    assert!(
        game.position()
            .stones()
            .iter()
            .all(|s| !rect.contains(s.spot))
    );
    assert!(list.candidates.iter().any(|c| {
        c.sources & JOIN != 0
            && c.protected
            && c.groups
                .iter()
                .filter(|&&i| geometry.groups[i].friendly)
                .count()
                == 3
    }));
    for c in list.candidates.iter().filter(|c| c.sources & JOIN != 0) {
        let analysis = game
            .position()
            .prepare()
            .analyze_move(
                Stone {
                    color: game.on_move(),
                    spot: c.spot(),
                },
                0,
            )
            .unwrap();
        assert!(analysis.connected_groups >= 2);
    }
}

#[test]
fn cell_rectangles_match_partition() {
    for resolution in [2, 38, 54, 78, 1002] {
        for id in [0, 55055, 500500, 999999] {
            let cell = cells::cell(id, resolution);
            let rect = cell_rect(cell, resolution);
            assert!(rect.contains(Spot {
                x: (id % 1000) as i32,
                y: (id / 1000) as i32
            }));
            for x in rect.left..=rect.right {
                assert_eq!(cells::cell((x + rect.top * 1000) as u32, resolution), cell);
            }
        }
    }
}

#[test]
fn capture_intersection_matches_exhaustive_strict_cover_and_blockers() {
    let game = Game::new(Board::new(9).unwrap())
        .play(Spot { x: 500, y: 500 })
        .unwrap();
    let liberties = [
        LibertySpan {
            y: 210,
            left: 220,
            right: 230,
        },
        LibertySpan {
            y: 211,
            left: 222,
            right: 228,
        },
    ];
    let mut free = FreeRows::new(&game);
    let actual = points(&capture_region(&liberties, game.board(), &mut free));
    let d = game.board().diameter();
    let mut expected = BTreeSet::new();
    for y in 100..330 {
        for x in 100..340 {
            let p = Spot { x, y };
            if game.position().can_place(p)
                && points(&liberties)
                    .iter()
                    .all(|&(lx, ly)| distance(p, Spot { x: lx, y: ly }) < i64::from(d).pow(2))
            {
                expected.insert((x, y));
            }
        }
    }
    assert_eq!(actual, expected);
    // An out-of-cell liberty must not be discarded when testing capture.
    let far = [
        liberties[0],
        LibertySpan {
            y: 700,
            left: 700,
            right: 700,
        },
    ];
    assert!(capture_region(&far, game.board(), &mut free).is_empty());
    // Exactly diameter distance is not covered by a strict exclusion disc.
    assert!(!actual.contains(&(220 - d, 210)));
}

#[test]
fn free_rows_match_authoritative_placement_geometry() {
    let game = Game::new(Board::new(9).unwrap())
        .play(Spot { x: 250, y: 250 })
        .unwrap()
        .play(Spot { x: 500, y: 500 })
        .unwrap();
    let rect = Rect {
        left: 140,
        right: 265,
        top: 140,
        bottom: 265,
    };
    let free = points(&FreeRows::new(&game).region(rect));
    for y in rect.top..=rect.bottom {
        for x in rect.left..=rect.right {
            assert_eq!(
                free.contains(&(x, y)),
                game.position().can_place(Spot { x, y })
            );
        }
    }
}

#[test]
fn retaining_geometry_preserves_features_and_source_group_identity() {
    let game = Game::new(Board::new(9).unwrap())
        .play(Spot { x: 300, y: 300 })
        .unwrap()
        .play(Spot { x: 600, y: 300 })
        .unwrap()
        .play(Spot { x: 450, y: 500 })
        .unwrap();
    let ordinary = FeatureExtractor::new(&game, 4).unwrap();
    assert!(ordinary.geometry.is_none());
    let (retained, _) = FeatureExtractor::with_geometry(&game, 4, 5, false).unwrap();
    assert_eq!(ordinary.coarse, retained.coarse);
    let geometry = retained.geometry.unwrap();
    let prepared = game.position().prepare();
    for g in &geometry.groups {
        assert_eq!(g.spans, prepared.liberty_spans(&g.members));
        assert_eq!(
            g.friendly,
            game.position().stones()[g.members[0]].color == game.on_move()
        );
    }
    assert!(geometry.bytes() > 0);
    assert!(FeatureExtractor::with_geometry(&game, 4, 4, false).is_err());
}

#[test]
fn deterministic_shortlists_respect_cells_protect_extrema_and_join() {
    for size in [3, 9, 13, 19] {
        let board = Board::new(size).unwrap();
        let d = board.diameter();
        let game = Game::new(board)
            .play(Spot { x: 500 - d, y: 400 })
            .unwrap()
            .play(Spot { x: 800, y: 800 })
            .unwrap()
            .play(Spot { x: 500 + d, y: 400 })
            .unwrap()
            .pass()
            .unwrap();
        let (extractor, _) = FeatureExtractor::with_geometry(&game, 4, 5, false).unwrap();
        let geometry = extractor.geometry.unwrap();
        let resolution = extractor.resolution;
        let middle = Spot { x: 500, y: 400 };
        let selected = [cells::cell((middle.x + 1000 * middle.y) as u32, resolution)];
        let a = generate(&game, &geometry, resolution, &selected, 8, 42).unwrap();
        let b = generate(&game, &geometry, resolution, &selected, 8, 42).unwrap();
        assert_eq!(a.candidates, b.candidates);
        assert!(a.profile.is_none());
        let started = Instant::now();
        let profiled = generate_profiled(&game, &geometry, resolution, &selected, 8, 42).unwrap();
        let total = started.elapsed().as_secs_f64();
        assert_eq!(a.candidates, profiled.candidates);
        let profile = profiled.profile.unwrap();
        assert!(profile.measured_seconds() > 0.0);
        assert!(profile.measured_seconds() <= total);
        for (i, bit) in [EXTEND, JOIN, HUG, CAPTURE, COVERAGE].iter().enumerate() {
            assert_eq!(
                profile.final_source_counts[i],
                a.candidates.iter().filter(|c| c.sources & bit != 0).count() as u64
            );
        }
        assert!(a.candidates.iter().any(|c| c.sources & JOIN != 0));
        assert_eq!(a.preliminary_candidates, a.candidates.len());
        assert!(a.candidates.iter().any(|c| c.sources & COVERAGE != 0));
        let rect = cell_rect(selected[0], resolution);
        let ids: BTreeSet<_> = a.candidates.iter().map(|c| c.action).collect();
        assert_eq!(ids.len(), a.candidates.len());
        for c in &a.candidates {
            assert_eq!(cells::cell(c.action, resolution), selected[0]);
            assert!(game.position().can_place(c.spot()));
            assert!(c.groups.windows(2).all(|w| w[0] < w[1]));
        }
        let mut reach = ReachRows::new(&game, &geometry);
        for (i, g) in geometry
            .groups
            .iter()
            .enumerate()
            .filter(|(_, g)| g.friendly)
        {
            let outer = reach.outer(i, &clip(&g.spans, rect));
            let extends: Vec<_> = a
                .candidates
                .iter()
                .filter(|c| c.extension_groups.contains(&i))
                .collect();
            assert!(extends.len() <= 3);
            for c in extends {
                assert!(c.sources & HUG != 0 || contains(&outer, c.spot()));
            }
        }
        assert_eq!(a.version, VERSION);
        assert_eq!(a.candidates.len().saturating_sub(8), a.protected_overflow);
    }
}

#[test]
fn corner_capture_has_validated_representatives_for_both_colours() {
    for white in [false, true] {
        let game = Game::new(Board::new(9).unwrap());
        let game = if white { game.pass().unwrap() } else { game };
        let game = game.play(Spot { x: 55, y: 55 }).unwrap();
        let (extractor, _) = FeatureExtractor::with_geometry(&game, 4, 5, false).unwrap();
        let geometry = extractor.geometry.unwrap();
        let enemy = geometry.groups.iter().find(|g| !g.friendly).unwrap();
        let witness = game
            .position()
            .prepare()
            .analyze_group(&enemy.members)
            .capture_in_one
            .unwrap();
        let cell = cells::cell((witness.x + 1000 * witness.y) as u32, extractor.resolution);
        let shortlist = generate(&game, &geometry, extractor.resolution, &[cell], 8, 0).unwrap();
        let captures: Vec<_> = shortlist
            .candidates
            .iter()
            .filter(|c| c.sources & CAPTURE != 0)
            .collect();
        assert!(!captures.is_empty());
        for c in captures {
            let child = game.play(c.spot()).unwrap();
            assert_eq!(child.position().stones().len(), 1);
            assert!(c.protected);
        }
    }
}

#[test]
fn connected_hug_is_kept_even_inside_maximum_reach() {
    let game = Game::new(Board::new(9).unwrap())
        .play(Spot { x: 300, y: 500 })
        .unwrap()
        .play(Spot { x: 520, y: 500 })
        .unwrap();
    let (input, _) = FeatureExtractor::with_geometry(&game, 4, 5, false).unwrap();
    let geometry = input.geometry.unwrap();
    let cell = cells::cell(500410, input.resolution);
    let list = generate(&game, &geometry, input.resolution, &[cell], 16, 0).unwrap();
    let hugs: Vec<_> = list
        .candidates
        .iter()
        .filter(|c| c.sources & HUG != 0)
        .collect();
    assert_eq!(hugs.len(), 1);
    let hug = hugs[0];
    assert!(hug.sources & EXTEND != 0 && !hug.frontier && hug.protected);
    assert!(game.board().adjacent(hug.spot(), Spot { x: 300, y: 500 }));
    assert!(DIRECTIONS.iter().any(|&(dx, dy)| game.board().overlaps(
        Spot {
            x: hug.spot().x + dx,
            y: hug.spot().y + dy
        },
        Spot { x: 520, y: 500 }
    )));
    assert_eq!(
        list.candidates
            .iter()
            .filter(|c| c.sources & EXTEND != 0)
            .count(),
        1
    );
    for c in &list.candidates {
        assert!(game.position().can_place(c.spot()));
    }
}

#[test]
fn extend_uses_real_outer_reach_not_cell_corners_or_overlap_edges() {
    let game = Game::new(Board::new(9).unwrap())
        .play(Spot { x: 300, y: 500 })
        .unwrap()
        .play(Spot { x: 800, y: 800 })
        .unwrap();
    let (input, _) = FeatureExtractor::with_geometry(&game, 4, 5, false).unwrap();
    let geometry = input.geometry.unwrap();
    let i = geometry.groups.iter().position(|g| g.friendly).unwrap();
    let cell = cells::cell(500450, input.resolution);
    let rect = cell_rect(cell, input.resolution);
    let local = clip(&geometry.groups[i].spans, rect);
    let expected: BTreeSet<_> = points(&local)
        .into_iter()
        .filter(|&(x, y)| {
            DIRECTIONS.iter().any(|&(dx, dy)| {
                !game.board().adjacent(
                    Spot {
                        x: x + dx,
                        y: y + dy,
                    },
                    Spot { x: 300, y: 500 },
                )
            })
        })
        .collect();
    let actual = ReachRows::new(&game, &geometry).outer(i, &local);
    assert_eq!(points(&actual), expected);
    assert!(!expected.is_empty());
    let list = generate(&game, &geometry, input.resolution, &[cell], 16, 0).unwrap();
    let extensions: Vec<_> = list
        .candidates
        .iter()
        .filter(|c| c.sources & EXTEND != 0)
        .collect();
    assert!(!extensions.is_empty());
    assert!(extensions.len() <= 3);
    assert!(extensions.iter().all(|c| c.frontier
        && c.sources & HUG == 0
        && expected.contains(&(c.spot().x, c.spot().y))));
    let maximum = expected
        .iter()
        .map(|&(x, y)| distance(Spot { x, y }, Spot { x: 300, y: 500 }))
        .max()
        .unwrap();
    assert!(
        extensions
            .iter()
            .any(|c| distance(c.spot(), Spot { x: 300, y: 500 }) == maximum)
    );
    // This cell is entirely inside connection reach: no artificial corner extrema.
    let interior = cells::cell(500410, input.resolution);
    let list = generate(&game, &geometry, input.resolution, &[interior], 16, 0).unwrap();
    assert!(list.candidates.iter().all(|c| c.sources & EXTEND == 0));
    assert!(list.candidates.len() <= 4);
}

#[test]
fn actual_board_limit_is_an_extension_frontier() {
    let game = Game::new(Board::new(9).unwrap())
        .play(Spot { x: 300, y: 200 })
        .unwrap()
        .play(Spot { x: 800, y: 800 })
        .unwrap();
    let (input, _) = FeatureExtractor::with_geometry(&game, 4, 5, false).unwrap();
    let geometry = input.geometry.unwrap();
    let cell = cells::cell(55300, input.resolution);
    let list = generate(&game, &geometry, input.resolution, &[cell], 16, 0).unwrap();
    assert!(
        list.candidates
            .iter()
            .any(|c| c.sources & EXTEND != 0 && c.frontier && c.spot().y == 55)
    );
}

#[test]
fn join_keeps_four_lens_aligned_extremes_including_rotated_side_tips() {
    for (a, b) in [
        (Spot { x: 370, y: 400 }, Spot { x: 630, y: 400 }),
        (Spot { x: 350, y: 350 }, Spot { x: 560, y: 560 }),
    ] {
        let game = Game::new(Board::new(9).unwrap())
            .play(a)
            .unwrap()
            .play(Spot { x: 900, y: 900 })
            .unwrap()
            .play(b)
            .unwrap()
            .pass()
            .unwrap();
        let (input, _) = FeatureExtractor::with_geometry(&game, 4, 5, false).unwrap();
        let geometry = input.geometry.unwrap();
        let friends: Vec<_> = geometry
            .groups
            .iter()
            .enumerate()
            .filter(|(_, g)| g.friendly)
            .map(|(i, _)| i)
            .collect();
        let region = intersect(
            &geometry.groups[friends[0]].spans,
            &geometry.groups[friends[1]].spans,
        );
        let (dx, dy) = (b.x - a.x, b.y - a.y);
        let directions = [(-dy, dx), (dy, -dx), (dx, dy), (-dx, -dy)];
        let origin = bounds(&region).unwrap().center();
        let tips: Vec<_> = directions
            .iter()
            .filter_map(|&v| extreme(&region, v, origin))
            .collect();
        let selected: Vec<_> = tips
            .iter()
            .map(|p| cells::cell((p.x + 1000 * p.y) as u32, input.resolution))
            .collect();
        let list = generate(&game, &geometry, input.resolution, &selected, 16, 0).unwrap();
        let mut expected = BTreeSet::new();
        let selected: BTreeSet<_> = selected.into_iter().collect();
        for cell in selected {
            let clipped = clip(&region, cell_rect(cell, input.resolution));
            let origin = bounds(&clipped).unwrap().center();
            for v in directions {
                let p = extreme(&clipped, v, origin).unwrap();
                expected.insert((p.x + 1000 * p.y) as u32);
            }
        }
        let got: BTreeSet<_> = list
            .candidates
            .iter()
            .filter(|c| c.sources & JOIN != 0)
            .map(|c| c.action)
            .collect();
        assert_eq!(got, expected); // No extra center or per-compass padding.
        for v in &directions[..2] {
            let projection =
                |p: Spot| i64::from(v.0) * i64::from(p.x) + i64::from(v.1) * i64::from(p.y);
            let maximum = points(&region)
                .into_iter()
                .map(|(x, y)| projection(Spot { x, y }))
                .max()
                .unwrap();
            assert_eq!(
                list.candidates
                    .iter()
                    .filter(|c| c.sources & JOIN != 0)
                    .map(|c| projection(c.spot()))
                    .max()
                    .unwrap(),
                maximum
            );
        }
    }
}
