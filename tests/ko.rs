use continugo_rules::{Action, Game, Play, Spot, Stone, sgf};
use continugo_train::{
    features::FeatureExtractor,
    replay::ReplayGame,
    search::{SearchConfig, search},
};
use rand::{SeedableRng, rngs::StdRng};

fn before_recapture() -> Game {
    Game::from_record(
        &sgf::decode("B[igjG];W[lgig];B[jGig];W[mGjG];B[jGlg];W[lglg];B[];W[jGjG];B[lgjG]")
            .unwrap(),
    )
    .unwrap()
}

#[test]
fn live_masks_and_search_forbid_exact_and_shifted_recaptures() {
    let game = before_recapture();
    let actions = [500_500, 501_500, 200_200].map(|id| Action::from_index(id).unwrap());
    for schema in [4, 5, 6, 7] {
        let extractor = FeatureExtractor::with_schema(&game, 1, schema).unwrap();
        let moves = extractor.moves(&actions).unwrap();
        assert_eq!(moves.legal, [0, 0, 1]);
        let fields = moves.exact.len() / actions.len();
        for row in moves.exact.chunks(fields).take(2) {
            assert!(row[..fields - 4].iter().all(|&v| v == 0.0));
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let parallel = extractor.moves_parallel(&actions, &pool).unwrap();
        assert_eq!(moves.legal, parallel.legal);
        assert_eq!(moves.exact, parallel.exact);
    }
    let mut rng = StdRng::seed_from_u64(42);
    let result = search(
        &game,
        &SearchConfig {
            simulations: 32,
            ..Default::default()
        },
        &mut rng,
    )
    .unwrap();
    for visit in result.policy {
        if let Some(spot) = visit.action.spot() {
            assert!(game.play(spot).is_ok());
        }
    }
    // A pass clears ko; the same position alone does not determine legality.
    let passed = game.pass().unwrap();
    assert_eq!(game.position(), passed.position());
    let recapture = Spot { x: 500, y: 500 };
    assert!(passed.play(recapture).is_ok());
}

#[test]
fn historical_recaptures_and_policy_visits_survive_replay() {
    let before = before_recapture();
    let stone = Stone {
        color: before.on_move(),
        spot: Spot { x: 500, y: 500 },
    };
    let mut record = sgf::Record {
        plays: before.record().to_vec(),
        ..Default::default()
    };
    record.plays.push(Play::Place(stone));
    record.move_properties = vec![vec![]; record.plays.len()];
    let finished = before
        .replay(Play::Place(stone))
        .unwrap()
        .pass()
        .unwrap()
        .pass()
        .unwrap();
    record.plays = finished.record().to_vec();
    record.move_properties.resize(record.plays.len(), vec![]);
    record.root.push((
        "RE".into(),
        vec![sgf::format_result(finished.board(), finished.result())],
    ));
    record.move_properties[9].push(("C".into(), vec![r#"{"policy":[["jGjG",7],["",1]]}"#.into()]));
    let text = sgf::encode_record(&record).unwrap();
    let replay = ReplayGame::load(&text).unwrap();
    assert_eq!(replay.info(1).unwrap().policy_visit_totals[9], Some(8));
    let sample = replay.position(9).unwrap();
    assert_eq!(sample.policy.unwrap()[0].visits, 7);
    assert!(sample.outcome.is_some());
    assert!(sample.final_targets.is_some());
    let after = replay.position(10).unwrap().game;
    assert!(after.position().stones().contains(&stone));
    assert!(after.play(Spot { x: 578, y: 500 }).is_err());
    assert_eq!(replay.viewer_position(9).unwrap()["action"], 500_500);
}
