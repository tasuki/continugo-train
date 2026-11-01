use continugo_rating::{GameResult, Rating};
use serde_json::{Value, json};
use std::{
    io::Write,
    process::{Command, Stdio},
};

fn run(request: Value) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_rating"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(request.to_string().as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn simultaneous_updates_keep_reference_and_inactive_bots_fixed() {
    let output = run(json!({
        "players": [
            {"id":"random", "rating":100.0, "deviation":0.0},
            {"id":"a", "rating":100.0, "deviation":350.0},
            {"id":"b", "rating":300.0, "deviation":100.0},
            {"id":"idle", "rating":175.0, "deviation":50.0}
        ],
        "games": [
            {"black":"a", "white":"random", "score":1.0},
            {"black":"a", "white":"b", "score":1.0}
        ]
    }));
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let response: Value = serde_json::from_slice(&output.stdout).unwrap();
    let players = response["players"].as_array().unwrap();
    assert_eq!(
        players[0],
        json!({"id":"random", "rating":100.0, "deviation":0.0})
    );
    assert_eq!(
        players[3],
        json!({"id":"idle", "rating":175.0, "deviation":50.0})
    );
    let expected = Rating {
        rating: 300.0,
        deviation: 100.0,
    }
    .update(&[GameResult {
        opponent: Rating::default(),
        score: 0.0,
    }])
    .unwrap();
    assert!((players[2]["rating"].as_f64().unwrap() - expected.rating).abs() < 1e-10);
    assert!(players[1]["deviation"].as_f64().unwrap() < 350.0);
}

#[test]
fn rejects_duplicate_players_self_matches_and_unknown_outcomes() {
    for request in [
        json!({"players":[{"id":"a", "rating":100, "deviation":350},
                          {"id":"a", "rating":100, "deviation":350}], "games":[]}),
        json!({"players":[{"id":"a", "rating":100, "deviation":350}],
               "games":[{"black":"a", "white":"a", "score":1}]}),
        json!({"players":[{"id":"a", "rating":100, "deviation":350},
                          {"id":"b", "rating":100, "deviation":350}],
               "games":[{"black":"a", "white":"b", "score":null}]}),
    ] {
        assert!(!run(request).status.success());
    }
}
