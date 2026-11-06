use continugo_rules::{Action, Board, Color, Game, Play, Spot, sgf, territory};
use continugo_train::{
    features::{COARSE_PLANES, FeatureExtractor, MOVE_FEATURES, count_feature},
    replay::load_position,
};

fn action(x: i32, y: i32) -> Action {
    Action::from_index((x + 1000 * y) as u32).unwrap()
}

#[test]
fn empty_positions_have_real_borders_and_legal_masks_across_sizes() {
    for (size, resolution) in [(9, 38), (13, 54), (19, 78)] {
        let game = Game::new(Board::new(size).unwrap());
        let features = FeatureExtractor::new(&game, 4).unwrap();
        assert_eq!(features.resolution, resolution);
        assert_eq!(
            features.coarse.len(),
            COARSE_PLANES.len() * resolution * resolution
        );
        assert!(
            features.coarse[..8 * resolution * resolution]
                .iter()
                .all(|&v| v == 0.0)
        );
        let actions = [action(0, 0), action(500, 500), action(999, 999)];
        let moves = features.moves(&actions).unwrap();
        assert_eq!(moves.legal, vec![0, 1, 0]);
        assert_eq!(moves.exact[MOVE_FEATURES.len() + 5], count_feature(1));
        assert!(features.moves(&[Action::PASS]).is_err());
    }
}

#[test]
fn profiled_coarse_features_preserve_bits_and_count_work() {
    for size in [9, 13, 19] {
        let board = Board::new(size).unwrap();
        let r = board.stone_radius();
        let mut game = Game::new(board);
        for spot in [Spot { x: r, y: r }, Spot { x: 500, y: 500 }] {
            game = game.play(spot).unwrap();
        }
        let ordinary = FeatureExtractor::new(&game, 4).unwrap();
        let (profiled, profile) = FeatureExtractor::profiled(&game, 4).unwrap();
        assert!(
            ordinary
                .coarse
                .iter()
                .map(|v| v.to_bits())
                .eq(profiled.coarse.iter().map(|v| v.to_bits()))
        );
        assert_eq!(profile.positions, 1);
        assert_eq!(profile.stones, 2);
        assert_eq!(profile.groups, 2);
        assert!(profile.spans > 0);
        assert_eq!(profile.preparation.calls, 1);
        assert_eq!(profile.liberty_spans.calls, 2);
        assert_eq!(profile.capturability.calls, 2);
        assert_eq!(profile.liberty_area.calls, 2);
        assert_eq!(profile.painting.calls, 4); // allocation, two groups, borders
        let mut total = continugo_train::features::CoarseProfile::default();
        total.merge(&profile);
        total.merge(&profile);
        assert_eq!(total.groups, 4);
        assert_eq!(
            total.liberty_area.seconds,
            2.0 * profile.liberty_area.seconds
        );
    }
    let finished = Game::default().pass().unwrap().pass().unwrap();
    assert!(FeatureExtractor::profiled(&finished, 4).is_err());
}

#[test]
fn parallel_move_features_match_serial_bits_and_action_order() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for size in [9, 13, 19] {
        let board = Board::new(size).unwrap();
        let r = board.stone_radius();
        let game = Game::new(board).play(Spot { x: r, y: r }).unwrap();
        let features = FeatureExtractor::new(&game, 4).unwrap();
        // Full domain once; noncanonical order, duplicates and uneven chunks too.
        let step = if size == 9 { 1 } else { 127 };
        let mut actions: Vec<_> = (0..1_000_000)
            .step_by(step)
            .map(|id| Action::from_index(id).unwrap())
            .collect();
        let group = game.position().groups(Color::Black).remove(0);
        let capture = game
            .position()
            .prepare()
            .analyze_group(&group)
            .capture_in_one
            .unwrap();
        actions.extend([action(r, r), action(r, r), action(capture.x, capture.y)]);
        actions.reverse();
        let serial = features.moves(&actions).unwrap();
        let parallel = features.moves_parallel(&actions, &pool).unwrap();
        assert_eq!(serial.legal, parallel.legal);
        assert!(
            serial
                .exact
                .iter()
                .map(|v| v.to_bits())
                .eq(parallel.exact.iter().map(|v| v.to_bits()))
        );
        assert!(features.moves_parallel(&[Action::PASS], &pool).is_err());
        assert!(
            features
                .moves_parallel(&[], &pool)
                .unwrap()
                .exact
                .is_empty()
        );
    }
}

#[test]
fn friendly_enemy_planes_are_side_to_move_relative() {
    let game = Game::default().play(Spot { x: 500, y: 500 }).unwrap();
    let white = FeatureExtractor::new(&game, 4).unwrap();
    let area = white.resolution.pow(2);
    assert!(white.coarse[..area].iter().all(|&v| v == 0.0));
    assert!(white.coarse[area..2 * area].contains(&1.0));
    let passed = game.pass().unwrap();
    let black = FeatureExtractor::new(&passed, 4).unwrap();
    assert!(black.coarse[..area].contains(&1.0));
    assert!(black.coarse[area..2 * area].iter().all(|&v| v == 0.0));
}

#[test]
fn one_move_capture_planes_use_legal_opposing_moves_not_liberty_counts() {
    for size in [9, 13, 19] {
        let board = Board::new(size).unwrap();
        let r = board.stone_radius();
        let corner = Game::new(board).play(Spot { x: r, y: r }).unwrap();
        let group = corner.position().groups(Color::Black).remove(0);
        assert_eq!(corner.position().group_liberties(&group, 8).len(), 8);
        let analysis = corner.position().prepare().analyze_group(&group);
        let capture = analysis
            .capture_in_one
            .expect("a diagonal placement captures a corner stone");
        let next = corner.play(capture).unwrap();
        assert!(
            next.position()
                .stones()
                .iter()
                .all(|s| s.color == Color::White)
        );
        let enemy = FeatureExtractor::new(&corner, 4).unwrap();
        let area = enemy.resolution.pow(2);
        assert_eq!(
            &enemy.coarse[area..2 * area],
            &enemy.coarse[7 * area..8 * area]
        );
        assert!(enemy.coarse[7 * area..8 * area].contains(&1.0));
        let passed = corner.pass().unwrap();
        let friendly = FeatureExtractor::new(&passed, 4).unwrap();
        assert_eq!(
            &friendly.coarse[..area],
            &friendly.coarse[6 * area..7 * area]
        );
        let center = Game::new(board).play(Spot { x: 500, y: 500 }).unwrap();
        let safe = FeatureExtractor::new(&center, 4).unwrap();
        assert!(safe.coarse[6 * area..8 * area].iter().all(|&x| x == 0.0));
    }
}

#[test]
fn features_do_not_depend_on_the_last_four_moves() {
    let mut first = Game::default();
    let mut second = Game::default();
    for (x, y) in [(200, 200), (600, 600), (400, 400), (800, 800)] {
        first = first.play(Spot { x, y }).unwrap();
    }
    for (x, y) in [(400, 400), (800, 800), (200, 200), (600, 600)] {
        second = second.play(Spot { x, y }).unwrap();
    }
    assert_ne!(first.record(), second.record());
    let first = FeatureExtractor::new(&first, 4).unwrap();
    let second = FeatureExtractor::new(&second, 4).unwrap();
    assert_eq!(first.coarse, second.coarse);
    let actions = [action(200, 200), action(276, 200), action(500, 500)];
    assert_eq!(
        first.moves(&actions).unwrap().exact,
        second.moves(&actions).unwrap().exact
    );
    assert!(
        !COARSE_PLANES
            .iter()
            .any(|name| name.contains("one_liberty"))
    );
    assert!(
        !MOVE_FEATURES
            .iter()
            .any(|name| name.contains("one_liberty"))
    );
}

#[test]
fn exact_features_distinguish_touching_connections_captures_and_suicide() {
    let game = Game::default()
        .play(Spot { x: 200, y: 200 })
        .unwrap()
        .pass()
        .unwrap();
    let features = FeatureExtractor::new(&game, 4).unwrap();
    let moves = features
        .moves(&[action(275, 200), action(276, 200), action(500, 500)])
        .unwrap();
    assert_eq!(moves.legal, vec![0, 1, 1]);
    assert_eq!(moves.exact[MOVE_FEATURES.len() + 1], count_feature(1));
    assert_eq!(moves.exact[MOVE_FEATURES.len() + 2], count_feature(1));
    assert_eq!(moves.exact[MOVE_FEATURES.len() + 5], count_feature(2));
    assert_eq!(moves.exact[2 * MOVE_FEATURES.len() + 1], 0.0);
    let fixture = include_str!("../../continugo/fixtures/rules/capture-four.txt");
    let record = fixture
        .lines()
        .find_map(|l| l.strip_prefix("record: "))
        .unwrap();
    let game = Game::from_record(&sgf::decode(record).unwrap()).unwrap();
    let features = FeatureExtractor::new(&game, 4).unwrap();
    let moves = features.moves(&[action(839, 240)]).unwrap();
    assert_eq!(moves.legal, vec![1]);
    assert_eq!(moves.exact[4], count_feature(4));
    assert_eq!(moves.exact[3], count_feature(1));
    let impossible = Game::new(Board::new(1).unwrap());
    assert_eq!(
        FeatureExtractor::new(&impossible, 4)
            .unwrap()
            .moves(&[action(500, 500)])
            .unwrap()
            .legal,
        vec![0]
    );
}

#[test]
fn the_bootstrap_eye_filter_is_not_a_neural_legality_rule() {
    let mut game = Game::new(Board::new(3).unwrap());
    for (x, y) in [
        (498, 166),
        (830, 166),
        (166, 498),
        (498, 498),
        (830, 498),
        (166, 830),
        (498, 830),
    ] {
        game = game.play(Spot { x, y }).unwrap().pass().unwrap();
    }
    let features = FeatureExtractor::new(&game, 4).unwrap();
    let area = features.resolution.pow(2);
    assert!(
        features.coarse[6 * area..8 * area]
            .iter()
            .all(|&x| x == 0.0)
    ); // Separate protected pockets.
    assert_eq!(features.moves(&[action(166, 166)]).unwrap().legal, vec![1]);
    let attack_position = game.play(Spot { x: 830, y: 830 }).unwrap();
    let attack_features = FeatureExtractor::new(&attack_position, 4).unwrap();
    assert!(attack_features.coarse[7 * area..8 * area].contains(&1.0));
    let attack = attack_features.moves(&[action(166, 166)]).unwrap();
    assert_eq!(attack.legal, vec![1]);
    assert_eq!(attack.exact[3], count_feature(1));
    assert_eq!(attack.exact[4], count_feature(8));
    let one_liberty = attack_position.pass().unwrap();
    let features = FeatureExtractor::new(&one_liberty, 4).unwrap();
    let area = features.resolution.pow(2);
    // One touching liberty center sweeps exactly one physical stone-disc.
    assert!(features.coarse[4 * area..5 * area].contains(&1.0));
    assert!(features.coarse[6 * area..7 * area].contains(&1.0)); // Opponent threat, even on the owner's turn.
    assert_eq!(features.moves(&[action(166, 166)]).unwrap().legal, vec![0]);
}

fn scored_record() -> sgf::Record {
    let game = Game::default()
        .play(Spot { x: 200, y: 200 })
        .unwrap()
        .pass()
        .unwrap()
        .pass()
        .unwrap();
    let result = territory::result(territory::areas(game.position().stones()));
    let mut record = sgf::Record {
        plays: game.record().to_vec(),
        ..Default::default()
    };
    record
        .root
        .push(("RE".into(), vec![sgf::format_result(game.board(), result)]));
    record.move_properties = vec![vec![]; record.plays.len()];
    record.move_properties[0].push(("C".into(), vec![r#"{"policy":[["dSdS",7],["",1]]}"#.into()]));
    record
}

#[test]
fn replay_aligns_pre_move_targets_values_and_natural_completion() {
    let mut record = scored_record();
    let sgf = sgf::encode_record(&record).unwrap();
    let first = load_position(&sgf, 0).unwrap();
    assert!(first.game.position().stones().is_empty());
    assert_eq!(first.outcome, Some(1.0));
    assert_eq!(first.policy.unwrap()[0].action, 200200);
    let second = load_position(&sgf, 1).unwrap();
    assert_eq!(second.game.on_move(), Color::White);
    assert_eq!(second.outcome, Some(-1.0));
    assert!(second.policy.is_none());
    assert!(load_position(&sgf, record.plays.len()).is_err());
    record.root.retain(|(name, _)| name != "RE");
    record.plays.truncate(1);
    record.move_properties.truncate(1);
    let truncated = sgf::encode_record(&record).unwrap();
    assert_eq!(load_position(&truncated, 0).unwrap().outcome, None);
    record.root.push(("RE".into(), vec!["B+1.0".into()]));
    assert!(load_position(&sgf::encode_record(&record).unwrap(), 0).is_err());
}

#[test]
fn replay_loads_generated_mcts_visits_without_inventing_labels() {
    use continugo_train::{
        search::SearchConfig,
        selfplay::{Engine, SelfplayConfig, generate},
    };
    let config = SelfplayConfig {
        engine: Engine::Mcts,
        max_moves: 512,
        search: SearchConfig {
            simulations: 8,
            ..Default::default()
        },
    };
    let record = generate(Board::new(9).unwrap(), &config, 41).unwrap();
    assert!(record.metadata.result.is_some());
    let loaded = load_position(&record.sgf, 0).unwrap();
    assert!(loaded.outcome.is_some());
    let policy = loaded.policy.unwrap();
    let original = record.metadata.policies[0].as_ref().unwrap();
    for target in &policy {
        let visit = original
            .iter()
            .find(|visit| visit.action.index() == target.action)
            .unwrap();
        assert_eq!(
            (target.action, target.visits),
            (visit.action.index(), visit.visits)
        );
    }
    assert_eq!(policy.iter().map(|t| t.visits).sum::<u32>(), 8);
}

#[test]
fn replay_decodes_policy_coordinates_and_pass_to_internal_action_ids() {
    let mut record = scored_record();
    record.move_properties[0][0].1 =
        vec![r#"{"policy":[["dSfO",7],["",1],["aMaM",0],["pass",0]]}"#.into()];
    let sample = load_position(&sgf::encode_record(&record).unwrap(), 0).unwrap();
    let targets: Vec<_> = sample
        .policy
        .unwrap()
        .into_iter()
        .map(|target| (target.action, target.visits))
        .collect();
    // "pass" is itself a valid four-letter coordinate, not a pass alias.
    assert_eq!(
        targets,
        [(300_200, 7), (1_000_000, 1), (38_038, 0), (954_780, 0)]
    );
}

#[test]
fn replay_accepts_only_compact_coordinate_visit_pairs() {
    for policy in [
        r#"[{"action":200200,"visits":1}]"#,
        r#"[[200200,1]]"#,
        r#"[["200200",1]]"#,
        r#"[["dS",1]]"#,
        r#"[["!!!!",1]]"#,
        r#"[["tmab",1]]"#, // x=1000 must not alias an in-domain action ID.
        r#"[["ZZZZ",1]]"#,
        r#"[["dSdS",-1]]"#,
        r#"[["dSdS",1.5]]"#,
        r#"[["dSdS",4294967296]]"#,
        r#"[["dSdS"]]"#,
        r#"[["dSdS",1,2]]"#,
        r#"[["dSdS",1],["dSdS",0]]"#,
        r#"[["",1],["",1]]"#,
        r#"[["dSdS",0]]"#,
        r#"[]"#,
    ] {
        let mut record = scored_record();
        record.move_properties[0][0].1 = vec![format!("{{\"policy\":{policy}}}")];
        assert!(
            load_position(&sgf::encode_record(&record).unwrap(), 0).is_err(),
            "accepted invalid policy: {policy}"
        );
    }
}

#[test]
fn display_evaluations_never_supply_or_override_training_labels() {
    let mut record = scored_record();
    record.move_properties[0][0].1 =
        vec![r#"{"win_pct":0.0,"score":-999.0,"policy":[["dSdS",7],["",1]]}"#.into()];
    let sample = load_position(&sgf::encode_record(&record).unwrap(), 0).unwrap();
    assert_eq!(sample.outcome, Some(1.0));
    assert_eq!(sample.final_targets.unwrap().score, 1.0);
    record.root.retain(|(name, _)| name != "RE");
    record.plays.truncate(1);
    record.move_properties.truncate(1);
    let sample = load_position(&sgf::encode_record(&record).unwrap(), 0).unwrap();
    assert!(sample.outcome.is_none());
    assert!(sample.final_targets.is_none());
}

#[test]
fn replay_rejects_bad_scores_illegal_targets_and_bad_future_moves() {
    let mut record = scored_record();
    record
        .root
        .iter_mut()
        .find(|(name, _)| name == "RE")
        .unwrap()
        .1 = vec!["W+1.0".into()];
    assert!(load_position(&sgf::encode_record(&record).unwrap(), 0).is_err());
    let mut record = scored_record();
    record.move_properties[0][0].1 = vec![r#"{"policy":[["aaaa",1]]}"#.into()];
    assert!(load_position(&sgf::encode_record(&record).unwrap(), 0).is_err());
    record.move_properties[0][0].1 = vec![r#"{"policy":[["dSdS",0]]}"#.into()];
    assert!(load_position(&sgf::encode_record(&record).unwrap(), 0).is_err());
    let unsupported = sgf::encode_record(&scored_record())
        .unwrap()
        .replace("SZ[13]", "SZ[13]AB[aaaa]");
    assert!(load_position(&unsupported, 0).is_err());
    record.plays[1] = Play::Pass(Color::Black);
    assert!(load_position(&sgf::encode_record(&record).unwrap(), 0).is_err());
}
