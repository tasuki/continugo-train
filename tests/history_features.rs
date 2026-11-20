use continugo_rules::{Action, Board, Color, Game, Spot};
use continugo_train::{
    cells,
    features::{COARSE_PLANES, FeatureExtractor, SCHEMA_VERSION},
};

fn marker(features: &FeatureExtractor<'_>, lag: usize, spot: Option<Spot>) {
    let area = features.resolution.pow(2);
    let plane = &features.coarse[(2 + lag) * area..(3 + lag) * area];
    assert!(plane.iter().all(|&v| (0.0..=1.0).contains(&v)));
    assert_eq!(plane.iter().any(|&v| v > 0.0), spot.is_some());
    if let Some(spot) = spot {
        assert!(plane[cells::cell((spot.x + 1000 * spot.y) as u32, features.resolution)] > 0.0);
        assert!(plane.iter().any(|&v| v > 0.0 && v < 1.0));
    }
}

#[test]
fn recent_moves_follow_occupancy_newest_first_and_preserve_schema6_bits() {
    assert_eq!(SCHEMA_VERSION, 7);
    assert_eq!(
        &COARSE_PLANES[..5],
        &[
            "friendly_stones",
            "enemy_stones",
            "enemy_move_1_ago_coverage",
            "friendly_move_2_ago_coverage",
            "enemy_move_3_ago_coverage",
        ]
    );
    for size in [9, 13, 19] {
        let board = Board::new(size).unwrap();
        let r = board.stone_radius();
        let spots = [
            Spot { x: r, y: r },
            Spot { x: 800, y: 800 },
            Spot { x: 500, y: 400 },
            Spot { x: 300, y: 700 },
        ];
        let mut game = Game::new(board);
        for ply in 0..=spots.len() {
            for density in [1, 4, 7] {
                let features = FeatureExtractor::new(&game, density).unwrap();
                let legacy = FeatureExtractor::with_schema(&game, density, 6).unwrap();
                let area = features.resolution.pow(2);
                assert_eq!(features.coarse.len(), 17 * area);
                assert_eq!(&features.coarse[..2 * area], &legacy.coarse[..2 * area]);
                assert_eq!(&features.coarse[5 * area..], &legacy.coarse[2 * area..]);
                for lag in 0..3 {
                    marker(&features, lag, ply.checked_sub(lag + 1).map(|i| spots[i]));
                    if ply > lag {
                        let color = game.record()[ply - lag - 1].color();
                        assert_eq!(
                            color,
                            if lag == 1 {
                                game.on_move()
                            } else {
                                game.on_move().other()
                            }
                        );
                    }
                }
                let actions = [
                    Action::from_index(500500).unwrap(),
                    Action::from_index(0).unwrap(),
                ];
                let a = features.moves(&actions).unwrap();
                let b = legacy.moves(&actions).unwrap();
                assert_eq!(a.exact, b.exact);
                assert_eq!(a.legal, b.legal);
            }
            if ply < spots.len() {
                game = game.play(spots[ply]).unwrap();
            }
        }
    }
}

#[test]
fn history_uses_occupancy_antialiasing_and_retains_subcell_motion() {
    for size in [9, 13, 19] {
        let board = Board::new(size).unwrap();
        for density in [1, 4, 7] {
            for spot in [
                Spot { x: 200, y: 201 },
                Spot {
                    x: board.stone_radius(),
                    y: board.stone_radius(),
                },
            ] {
                let game = Game::new(board).play(spot).unwrap();
                let features = FeatureExtractor::new(&game, density).unwrap();
                let area = features.resolution.pow(2);
                let history = &features.coarse[2 * area..3 * area];
                assert_eq!(history, &features.coarse[area..2 * area]);
                let edges = continugo_train::targets::raster_bounds(features.resolution);
                let mut covered = 0.0;
                for y in 0..features.resolution {
                    for x in 0..features.resolution {
                        covered += f64::from(history[y * features.resolution + x])
                            * (edges[x + 1] - edges[x])
                            * (edges[y + 1] - edges[y]);
                    }
                }
                let expected = std::f64::consts::PI * f64::from(board.stone_radius()).powi(2);
                assert!((covered - expected).abs() < 0.02);
            }
        }
        let game = Game::new(board).play(Spot { x: 200, y: 200 }).unwrap();
        let moved = Game::new(board).play(Spot { x: 201, y: 200 }).unwrap();
        let first = FeatureExtractor::new(&game, 4).unwrap();
        let second = FeatureExtractor::new(&moved, 4).unwrap();
        let area = first.resolution.pow(2);
        assert_eq!(
            cells::cell(200200, first.resolution),
            cells::cell(200201, first.resolution)
        );
        assert_ne!(
            &first.coarse[2 * area..3 * area],
            &second.coarse[2 * area..3 * area]
        );
    }
}

#[test]
fn passes_consume_slots_and_missing_history_is_blank() {
    let first = Spot { x: 200, y: 200 };
    let second = Spot { x: 700, y: 700 };
    let initial = Game::default();
    let opening_pass = initial.pass().unwrap();
    for game in [&initial, &opening_pass] {
        let features = FeatureExtractor::new(game, 4).unwrap();
        for lag in 0..3 {
            marker(&features, lag, None);
        }
    }
    let game = initial
        .play(first)
        .unwrap()
        .play(second)
        .unwrap()
        .pass()
        .unwrap();
    assert_eq!(game.on_move(), Color::White);
    let features = FeatureExtractor::new(&game, 4).unwrap();
    marker(&features, 0, None);
    marker(&features, 1, Some(second));
    marker(&features, 2, Some(first));
    let game = game.play(Spot { x: 500, y: 500 }).unwrap();
    let features = FeatureExtractor::new(&game, 4).unwrap();
    marker(&features, 0, Some(Spot { x: 500, y: 500 }));
    marker(&features, 1, None);
    marker(&features, 2, Some(second));
    assert!(FeatureExtractor::new(&initial.pass().unwrap().pass().unwrap(), 4).is_err());
}

#[test]
fn captured_placements_remain_in_history_and_geometry_retention_is_unchanged() {
    for size in [9, 13, 19] {
        let board = Board::new(size).unwrap();
        let r = board.stone_radius();
        let corner = Spot { x: r, y: r };
        let game = Game::new(board).play(corner).unwrap();
        let group = game.position().groups(Color::Black).remove(0);
        let capture = game
            .position()
            .prepare()
            .analyze_group(&group)
            .capture_in_one
            .unwrap();
        let game = game.play(capture).unwrap();
        assert!(
            game.position()
                .stones()
                .iter()
                .all(|s| s.color == Color::White)
        );
        let (features, _) = FeatureExtractor::with_geometry(&game, 4, 7, false).unwrap();
        marker(&features, 0, Some(capture));
        marker(&features, 1, Some(corner));
        marker(&features, 2, None);
        assert_eq!(features.geometry.as_ref().unwrap().groups.len(), 1);
        assert_eq!(
            features.coarse,
            FeatureExtractor::new(&game, 4).unwrap().coarse
        );
    }
}
