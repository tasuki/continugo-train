use continugo_rules::{Board, Game, sgf};
use serde_json::Value;
use std::{fs, path::PathBuf, process::Command};

fn cli() -> Command {
    Command::new(env!("CARGO_BIN_EXE_continugo-train"))
}

fn board_from_sgf(record: &sgf::Record) -> Board {
    let size = record.root.iter().find(|(name, _)| name == "SZ").unwrap().1[0]
        .parse()
        .unwrap();
    Board::new(size).unwrap()
}

fn comment(properties: &[sgf::Property]) -> Value {
    let values = &properties.iter().find(|(name, _)| name == "C").unwrap().1;
    assert_eq!(values.len(), 1);
    serde_json::from_str(&values[0]).unwrap()
}

struct TestDir(PathBuf);

impl TestDir {
    fn new(name: &str) -> Self {
        let path =
            std::env::temp_dir().join(format!("continugo-train-{name}-{}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn default_mix_and_truncation_warnings_are_explicit() {
    let directory = TestDir::new("weighted-default");
    let output = cli()
        .args([
            "--engine",
            "random",
            "--games",
            "5",
            "--max-moves",
            "1",
            "--output",
        ])
        .arg(&directory.0)
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output);
    let mut sizes = std::collections::BTreeMap::new();
    for entry in fs::read_dir(&directory.0).unwrap() {
        let record =
            sgf::decode_record(&fs::read_to_string(entry.unwrap().path()).unwrap()).unwrap();
        *sizes.entry(board_from_sgf(&record).size()).or_insert(0) += 1;
    }
    assert_eq!(sizes, [(9, 2), (13, 2), (19, 1)].into_iter().collect());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert_eq!(stderr.matches("WARNING TRUNCATED").count(), 5);
    assert!(stderr.contains("draws=0 truncated=5"));
}

#[test]
fn structured_run_stats_report_truncation_without_overwriting_evidence() {
    let directory = TestDir::new("stats-json");
    let path = directory.0.join("stats.json");
    let run = || {
        cli()
            .args([
                "--engine",
                "random",
                "--sizes",
                "9",
                "--max-moves",
                "1",
                "--stats-json",
            ])
            .arg(&path)
            .output()
            .unwrap()
    };
    let output = run();
    assert!(output.status.success(), "{:?}", output);
    let bytes = fs::read(&path).unwrap();
    let stats: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(stats["games"], 1);
    assert_eq!(stats["completed"], 0);
    assert_eq!(stats["truncated"], 1);
    assert_eq!(stats["moves"], 1);
    assert!(stats["seconds"].as_f64().unwrap() > 0.0);
    assert!(stats["inference"].is_null());
    assert!(!run().status.success());
    assert_eq!(fs::read(&path).unwrap(), bytes);
}

#[test]
fn stdout_previews_use_the_main_repo_sgf_codec() {
    for size in [9, 13, 19] {
        let output = cli()
            .args([
                "--engine",
                "random",
                "--sizes",
                &size.to_string(),
                "--max-moves",
                "2",
            ])
            .output()
            .unwrap();
        assert!(output.status.success(), "{:?}", output);
        let text = String::from_utf8(output.stdout).unwrap();
        let record = sgf::decode_record(&text).unwrap();
        let board = board_from_sgf(&record);
        assert_eq!(board.size(), size);
        assert_eq!(record.plays.len(), 2);
        assert_eq!(comment(&record.root), serde_json::json!({"seed": 0}));
        for name in ["PB", "PW"] {
            assert_eq!(
                record.root.iter().find(|(key, _)| key == name).unwrap().1,
                vec!["random"]
            );
        }
        assert!(!record.root.iter().any(|(key, _)| key == "AP"));
        assert!(record.move_properties.iter().all(|properties| {
            comment(properties) == serde_json::json!({"win_pct":null,"score":null})
        }));
        assert!(
            record
                .plays
                .iter()
                .all(|play| matches!(play, continugo_rules::Play::Place(_)))
        );
        assert!(
            !Game::from_record_on(board, &record.plays)
                .unwrap()
                .is_finished()
        );
    }
}

#[test]
fn saved_sgf_replays_and_metadata_preserves_sparse_visits_and_truncation() {
    let directory = TestDir::new("mcts-record");
    let output = cli()
        .args([
            "--max-moves",
            "1",
            "--simulations",
            "8",
            "--sizes",
            "19",
            "--output",
        ])
        .arg(&directory.0)
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output);
    assert!(output.stdout.is_empty());
    let text = fs::read_to_string(directory.0.join("game-000001-size19-seed0.sgf")).unwrap();
    let record = sgf::decode_record(&text).unwrap();
    assert_eq!(sgf::encode_record(&record).unwrap(), text);
    let board = board_from_sgf(&record);
    assert_eq!(board.size(), 19);
    let plays = record.plays;
    let game = Game::from_record_on(board, &plays).unwrap();
    assert_eq!(game.record().len(), 1);
    assert!(!game.is_finished());
    let metadata = comment(&record.root);
    assert_eq!(metadata["seed"], 0);
    assert_eq!(metadata["search"]["simulations"], 8);
    assert_eq!(metadata.as_object().unwrap().len(), 2);
    for name in ["PB", "PW"] {
        assert_eq!(
            record.root.iter().find(|(key, _)| key == name).unwrap().1,
            vec!["mcts"]
        );
    }
    assert!(!record.root.iter().any(|(name, _)| name == "RE"));
    let node = comment(&record.move_properties[0]);
    let policy = node["policy"].as_array().unwrap();
    assert_eq!(node.get("win_pct"), Some(&serde_json::Value::Null));
    assert_eq!(node.get("score"), Some(&serde_json::Value::Null));
    assert!(node.get("profile").is_none());
    assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 1);
    assert!(policy.iter().all(|entry| {
        let pair = entry.as_array().unwrap();
        pair.len() == 2 && pair[0].is_string() && pair[1].is_u64()
    }));
    assert!(
        policy
            .windows(2)
            .all(|pair| pair[0][1].as_u64() >= pair[1][1].as_u64())
    );
    assert_eq!(
        policy
            .iter()
            .map(|entry| entry[1].as_u64().unwrap())
            .sum::<u64>(),
        8
    );
}

#[test]
fn saved_random_games_do_not_pass_prematurely() {
    let directory = TestDir::new("completed-records");
    let output = cli()
        .args([
            "--engine",
            "random",
            "--games",
            "3",
            "--seed",
            "42",
            "--max-moves",
            "2",
            "--output",
        ])
        .arg(&directory.0)
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output);
    for (index, size) in [9, 13, 19].iter().enumerate() {
        let stem = format!("game-{:06}-size{size}-seed{}", index + 1, 42 + index);
        let text = fs::read_to_string(directory.0.join(format!("{stem}.sgf"))).unwrap();
        let record = sgf::decode_record(&text).unwrap();
        let board = board_from_sgf(&record);
        assert_eq!(board.size(), *size);
        assert_eq!(record.plays.len(), 2);
        assert!(
            record
                .plays
                .iter()
                .all(|play| matches!(play, continugo_rules::Play::Place(_)))
        );
        assert!(
            !Game::from_record_on(board, &record.plays)
                .unwrap()
                .is_finished()
        );
        let metadata = comment(&record.root);
        assert_eq!(metadata, serde_json::json!({"seed": 42 + index}));
        assert!(record.move_properties.iter().all(|properties| {
            comment(properties) == serde_json::json!({"win_pct":null,"score":null})
        }));
        assert!(!record.root.iter().any(|(name, _)| name == "RE"));
    }
    assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 3);
}

#[test]
fn saved_completed_mcts_game_has_a_scored_sgf_result() {
    let directory = TestDir::new("completed-mcts");
    let output = cli()
        .args([
            "--sizes",
            "19",
            "--seed",
            "1",
            "--simulations",
            "16",
            "--max-moves",
            "512",
            "--output",
        ])
        .arg(&directory.0)
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output);
    let text = fs::read_to_string(directory.0.join("game-000001-size19-seed1.sgf")).unwrap();
    let record = sgf::decode_record(&text).unwrap();
    let game = Game::from_record_on(board_from_sgf(&record), &record.plays).unwrap();
    assert!(game.is_finished());
    let metadata = comment(&record.root);
    let result = &record.root.iter().find(|(name, _)| name == "RE").unwrap().1[0];
    use continugo_rules::{Color, territory};
    match territory::result(territory::areas(game.position().stones())) {
        territory::Outcome::Draw => assert_eq!(result, "0"),
        territory::Outcome::Win(color, margin) => {
            let (winner, score) = result.split_once('+').unwrap();
            assert_eq!(winner, if color == Color::Black { "B" } else { "W" });
            let stone_area = std::f64::consts::PI * f64::from(game.board().stone_radius()).powi(2);
            assert_eq!(score.split_once('.').unwrap().1.len(), 1);
            assert!((score.parse::<f64>().unwrap() - margin / stone_area).abs() <= 0.05 + 1e-9);
        }
    }
    assert_eq!(record.move_properties.len(), record.plays.len());
    assert!(
        record
            .move_properties
            .iter()
            .all(|properties| comment(properties)["policy"].is_array())
    );
    assert_eq!(metadata["seed"], 1);
    assert_eq!(metadata["search"]["simulations"], 16);
    assert_eq!(metadata.as_object().unwrap().len(), 2);
    let terminal = comment(record.move_properties.last().unwrap());
    let mover = game.on_move().other();
    let (win_pct, score) = match territory::result(territory::areas(game.position().stones())) {
        territory::Outcome::Draw => (50.0, 0.0),
        territory::Outcome::Win(winner, margin) => {
            let stone_area =
                std::f64::consts::PI * f64::from(board_from_sgf(&record).stone_radius()).powi(2);
            let margin = (margin / stone_area * 10.0).round() / 10.0;
            if winner == mover {
                (100.0, margin)
            } else {
                (0.0, -margin)
            }
        }
    };
    assert_eq!(terminal["win_pct"], win_pct);
    assert_eq!(terminal["score"], score);
}

#[test]
fn bootstrap_guidance_is_explicit_recorded_and_forbidden_for_other_engines() {
    let output = cli()
        .args([
            "--engine",
            "mcts",
            "--sizes",
            "9",
            "--simulations",
            "4",
            "--max-moves",
            "1",
            "--bootstrap-avoid-eyes",
        ])
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let text = String::from_utf8(output.stdout).unwrap();
    let record = sgf::decode_record(text.trim()).unwrap();
    assert_eq!(
        comment(&record.root)["search"]["bootstrap_avoid_eyes"],
        true
    );
    assert!(!record.root.iter().any(|(name, _)| name == "RE"));
    let info = continugo_train::replay::ReplayGame::load(&text)
        .unwrap()
        .info(4)
        .unwrap();
    assert_eq!(info.policy_visit_totals, vec![Some(4)]);
    assert!(info.eligible_plies.is_empty());
    for engine in ["random", "neural"] {
        let output = cli()
            .args(["--engine", engine, "--bootstrap-avoid-eyes"])
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("requires --engine mcts"));
    }
}

#[test]
fn rollout_diagnostics_never_turn_an_unfinished_actual_game_into_a_label() {
    let directory = TestDir::new("rollout-censoring");
    let output = cli()
        .args([
            "--engine",
            "mcts",
            "--sizes",
            "9",
            "--games",
            "1",
            "--simulations",
            "8",
            "--max-moves",
            "1",
            "--rollouts",
            "2",
            "--rollout-max-moves",
            "1",
            "--rollout-pass-probability",
            "0",
            "--score-weight",
            "0",
            "--output",
        ])
        .arg(&directory.0)
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let text = fs::read_to_string(directory.0.join("game-000001-size9-seed0.sgf")).unwrap();
    let record = sgf::decode_record(&text).unwrap();
    let metadata = comment(&record.root);
    assert_eq!(metadata["search"]["rollout"]["playouts"], 2);
    assert_eq!(metadata["rollout_stats"]["completed"], 0);
    assert_eq!(metadata["rollout_stats"]["draws"], 0);
    assert!(metadata["rollout_stats"]["truncated"].as_u64().unwrap() > 0);
    assert!(!record.root.iter().any(|(name, _)| name == "RE"));
    let replay = continugo_train::replay::ReplayGame::load(&text).unwrap();
    assert!(replay.info(4).unwrap().eligible_plies.is_empty());
}

#[test]
fn unguided_reference_diagnostics_need_no_neural_runtime_and_keep_random_turns_unlabeled() {
    let directory = TestDir::new("unguided-reference");
    let output = Command::new(env!("CARGO_BIN_EXE_arena"))
        .args([
            "--unguided",
            "--reference-random",
            "--games",
            "2",
            "--sizes",
            "9",
            "--simulations",
            "8",
            "--max-moves",
            "2",
            "--ort-library",
            "/missing/unused.so",
            "--output",
        ])
        .arg(&directory.0)
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    for entry in fs::read_dir(&directory.0).unwrap() {
        let text = fs::read_to_string(entry.unwrap().path()).unwrap();
        let record = sgf::decode_record(&text).unwrap();
        let players: Vec<_> = ["PB", "PW"]
            .iter()
            .map(|key| record.root.iter().find(|(name, _)| name == key).unwrap().1[0].as_str())
            .collect();
        assert!(players.contains(&"mcts"));
        assert!(players.contains(&"random-no-eye-fill-v1"));
        assert!(record.root.iter().any(|(name, _)| name == "C"));
        for (index, properties) in record.move_properties.iter().enumerate() {
            let node = comment(properties);
            assert!(node.get("win_pct").is_some());
            assert!(node.get("score").is_some());
            assert_eq!(node.get("policy").is_some(), players[index % 2] == "mcts");
        }
        assert!(
            continugo_train::replay::ReplayGame::load(&text)
                .unwrap()
                .info(4)
                .unwrap()
                .eligible_plies
                .is_empty()
        );
    }
}

#[test]
fn random_profiling_is_reported_and_saved_in_metadata() {
    let directory = TestDir::new("profile");
    let output = cli()
        .args([
            "--engine",
            "random",
            "--profile",
            "--max-moves",
            "4",
            "--output",
        ])
        .arg(&directory.0)
        .output()
        .unwrap();
    assert!(output.status.success(), "{:?}", output);
    let text = fs::read_to_string(directory.0.join("game-000001-size9-seed0.sgf")).unwrap();
    let record = sgf::decode_record(&text).unwrap();
    assert_eq!(sgf::encode_record(&record).unwrap(), text);
    let metadata = comment(&record.root);
    let profile = &metadata["profile"];
    assert!(profile["selfplay_seconds"].as_f64().unwrap() > 0.0);
    assert!(profile.get("moves").is_none());
    assert!(profile.get("seconds").is_none());
    assert!(profile.get("sgf_seconds").is_none());
    assert_eq!(record.move_properties.len(), 4);
    for (index, properties) in record.move_properties.iter().enumerate() {
        let node = comment(properties);
        assert!(node.get("policy").is_none());
        assert_eq!(node["profile"]["ply"], index + 1);
        assert_eq!(
            node["profile"]["rules"]["calls"],
            node["profile"]["proposals"]
        );
    }
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("generation="));
    assert!(stderr.contains("output="));
    assert!(stderr.contains("rules / move application"));
    assert!(stderr.contains("small-eye detection"));
}

#[test]
fn invalid_settings_fail_and_existing_sgf_is_not_overwritten() {
    for args in [
        vec!["--sizes", "8"],
        vec!["--max-moves", "0"],
        vec!["--simulations", "0"],
        vec!["--proposal-attempts", "0"],
        vec!["--profile"],
        vec!["--rollouts", "1", "--rollout-max-moves", "0"],
        vec!["--rollouts", "1", "--rollout-pass-probability", "2"],
        vec!["--engine", "random", "--rollouts", "1"],
        vec!["--concurrent-games", "0"],
        vec!["--concurrent-games", "65"],
        vec!["--concurrent-games", "2"], // neural only
        vec!["--inference-batch-size", "0"],
        vec!["--inference-batch-size", "33"],
        vec!["--inference-batch-wait-ms", "1001"],
    ] {
        let output = cli().args(args).output().unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
    }
    let directory = TestDir::new("collisions");
    let sgf_path = directory.0.join("game-000001-size9-seed0.sgf");
    fs::write(&sgf_path, "keep existing data").unwrap();
    let output = cli()
        .args(["--max-moves", "1", "--output"])
        .arg(&directory.0)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert_eq!(fs::read_to_string(sgf_path).unwrap(), "keep existing data");
    assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 1);
}
