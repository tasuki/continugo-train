//! Explicit artifact-backed diagnostic; never writes models or campaign files.
use continugo_rules::{Board, Game, Spot};
use continugo_train::inference::{Evaluator, InferenceDevice, InferenceOutput, OnnxEvaluator};
use std::path::PathBuf;

#[test]
#[ignore = "set CONTINUGO_GEOMETRY_MODEL and ORT_DYLIB_PATH to existing schema-5 artifacts"]
fn geometry_policy_preserves_pass_cache_batching_and_value_only_semantics() {
    let model = PathBuf::from(std::env::var_os("CONTINUGO_GEOMETRY_MODEL").unwrap());
    let mut evaluator = OnnxEvaluator::load(&model, None, 1, InferenceDevice::Cpu).unwrap();
    evaluator.configure_features(1, 4).unwrap();
    let game = Game::new(Board::new(9).unwrap())
        .play(Spot { x: 300, y: 400 })
        .unwrap()
        .play(Spot { x: 600, y: 400 })
        .unwrap();
    assert_eq!(
        evaluator.geometry_budget(),
        Some(continugo_train::geometry_proposals::DEFAULT_BUDGET)
    );
    assert_eq!(
        evaluator.proposal_policy(),
        continugo_train::geometry_proposals::VERSION
    );
    let default = evaluator.evaluate(&game).unwrap();
    let default_rows = evaluator.stats.exact_coordinates;
    evaluator.configure_geometry(None).unwrap();
    assert_eq!(evaluator.proposal_policy(), "complete-cells-v1");
    evaluator.stats = Default::default();
    let baseline = evaluator.evaluate(&game).unwrap();
    let full_rows = evaluator.stats.exact_coordinates;
    assert!(default_rows < full_rows / 2);
    assert_eq!(default.value, baseline.value);
    assert_eq!(default.score, baseline.score);
    assert!(evaluator.configure_geometry(Some(0)).is_err());
    evaluator.configure_geometry(Some(16)).unwrap();
    assert_eq!(evaluator.stats.cache_entries, 0);
    evaluator.stats = Default::default();
    let estimate = evaluator.estimate(&game).unwrap();
    assert_eq!(evaluator.stats.exact_coordinates, 0);
    assert_eq!(evaluator.stats.proposal_preliminary, 0);
    assert_eq!(estimate.value, baseline.value);
    let a = evaluator.evaluate_seeded(&game, 42).unwrap();
    let first_rows = evaluator.stats.exact_coordinates;
    assert!(first_rows < full_rows / 2);
    let first_preliminary = evaluator.stats.proposal_preliminary;
    let b = evaluator.evaluate_seeded(&game, 42).unwrap();
    assert_eq!(a.logits, b.logits);
    assert_eq!(a.legal, b.legal);
    assert_eq!(a.logits[1_000_000], baseline.logits[1_000_000]);
    let mass: f64 = a.logits.iter().map(|&p| f64::from(p).exp()).sum();
    assert!((mass - 1.0).abs() < 1e-6);
    let shortlist = evaluator.geometry_shortlist(&game, 42).unwrap();
    assert_eq!(shortlist.candidates.len() as u64, first_rows);
    assert_eq!(shortlist.preliminary_candidates as u64, first_preliminary);
    assert_eq!(evaluator.stats.proposal_stages.measured_seconds(), 0.0);
    evaluator.profile_coarse = true;
    let profiled = evaluator.evaluate_seeded(&game, 42).unwrap();
    assert_eq!(profiled.logits, a.logits);
    assert_eq!(profiled.legal, a.legal);
    assert!(evaluator.stats.proposal_stages.measured_seconds() > 0.0);
    let selected = &shortlist.selected_cells;
    let resolution = continugo_train::features::resolution(
        game.board().diameter(),
        evaluator.samples_per_diameter(),
    )
    .unwrap();
    for (i, &legal) in a.legal.iter().enumerate() {
        if legal != 0 {
            assert!(selected.contains(&continugo_train::cells::cell(i as u32, resolution)));
            game.play(Spot {
                x: (i % 1000) as i32,
                y: (i / 1000) as i32,
            })
            .unwrap();
        }
    }
    assert_eq!(evaluator.stats.encoder_positions, 1); // estimate -> policy reuses geometry
    evaluator.configure_features(4, 4).unwrap();
    let batch = evaluator
        .evaluate_batch(&[(&game, true), (&game, true), (&game, false)])
        .unwrap();
    let [
        InferenceOutput::Policy(x),
        InferenceOutput::Policy(y),
        InferenceOutput::Value(v),
    ] = batch.as_slice()
    else {
        panic!("wrong batch kinds")
    };
    assert_eq!(x.logits, y.logits);
    assert_eq!(v.value, x.value);
    let serial = evaluator.evaluate(&game).unwrap();
    assert_eq!(x.logits, serial.logits);
    evaluator.configure_geometry(None).unwrap();
    assert_eq!(evaluator.stats.cache_entries, 0);
    let restored = evaluator.evaluate(&game).unwrap();
    assert_eq!(restored.logits, baseline.logits);
    evaluator
        .configure_geometry(Some(continugo_train::geometry_proposals::DEFAULT_BUDGET))
        .unwrap();
    let restored_default = evaluator.evaluate(&game).unwrap();
    assert_eq!(restored_default.logits, default.logits);
    assert_eq!(restored_default.legal, default.legal);
}
