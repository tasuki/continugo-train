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
        assert!(record.move_properties.iter().all(Vec::is_empty));
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
    assert!(node.get("profile").is_none());
    assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 1);
    assert_eq!(
        policy
            .iter()
            .map(|entry| entry["visits"].as_u64().unwrap())
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
        assert!(record.move_properties.iter().all(Vec::is_empty));
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
            "12",
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
