//! Native integration exercised by tests/test_inference.py with its exported model.
use continugo_rules::{Board, Game, Spot};
use continugo_train::inference::{
    Evaluator, InferenceDevice, InferenceOutput, OnnxEvaluator,
    batching::{BatchingConfig, BatchingWorker},
};
use std::{path::PathBuf, sync::Barrier, time::Duration};

fn close(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-5 * (1.0 + a.abs()), "{a} != {b}");
}

#[test]
#[ignore = "requires CONTINUGO_TEST_MODEL and ORT_DYLIB_PATH; run by Python inference tests"]
fn native_batch_parity_and_worker_lifecycle() {
    let model = PathBuf::from(std::env::var_os("CONTINUGO_TEST_MODEL").unwrap());
    let device = if std::env::var("CONTINUGO_TEST_DEVICE").as_deref() == Ok("cuda") {
        InferenceDevice::Cuda
    } else {
        InferenceDevice::Cpu
    };
    let mut evaluator = OnnxEvaluator::load(&model, None, 1, device).unwrap();
    evaluator.configure_features(4, 64).unwrap();
    let mut games = Vec::new();
    for size in [9, 13, 19] {
        let game = Game::new(Board::new(size).unwrap())
            .play(Spot { x: 200, y: 200 })
            .unwrap();
        games.push(game.clone());
        games.push(game.play(Spot { x: 500, y: 500 }).unwrap());
    }
    let singles: Vec<_> = games
        .iter()
        .map(|g| evaluator.evaluate(g).unwrap())
        .collect();
    evaluator.configure_features(4, 64).unwrap();
    evaluator.stats = Default::default();
    evaluator.profile_coarse = true;
    let queries: Vec<_> = games
        .iter()
        .map(|g| (g, true))
        .chain(games.iter().map(|g| (g, false)))
        .collect();
    let batch = evaluator.evaluate_batch(&queries).unwrap();
    assert_eq!(evaluator.stats.encoder_positions, 6); // duplicate queries coalesced
    assert_eq!(evaluator.stats.coarse_stages.positions, 6);
    assert_eq!(evaluator.stats.coarse_stages.stones, 9);
    assert_eq!(evaluator.stats.coarse_stages.preparation.calls, 6);
    assert_eq!(evaluator.stats.coarse_stages.liberty_spans.calls, 9);
    assert_eq!(evaluator.stats.coarse_stages.capturability.calls, 9);
    assert_eq!(evaluator.stats.encoder_max_batch, 2);
    assert_eq!(evaluator.stats.scorer_max_batch, 2);
    assert_eq!(evaluator.stats.calls, 6);
    assert_eq!(evaluator.stats.value_calls, 6);
    for ((actual, expected), value) in batch[..6].iter().zip(&singles).zip(&batch[6..]) {
        let InferenceOutput::Policy(actual) = actual else {
            panic!("policy expected")
        };
        let InferenceOutput::Value(value) = value else {
            panic!("value expected")
        };
        close(actual.value, expected.value);
        close(value.value, expected.value);
        close(value.score, expected.score);
        close(actual.score, expected.score);
        assert_eq!(actual.legal, expected.legal);
        assert_eq!(actual.logits.len(), 1_000_001);
        for (&a, &b) in actual.logits.iter().zip(&expected.logits) {
            if a.is_finite() {
                close(f64::from(a), f64::from(b));
            } else {
                assert_eq!(a, b);
            }
        }
    }
    // An odd final coordinate chunk and the aggregate-row cap (B > 4) must
    // preserve every action's row, including neighboring/chunk-boundary IDs.
    let wide_queries: Vec<_> = (0..6).map(|i| (&games[i % 2], true)).collect();
    let wide = evaluator.evaluate_batch(&wide_queries).unwrap();
    assert_eq!(evaluator.stats.scorer_max_batch, 6);
    for (i, actual) in wide.iter().enumerate() {
        let InferenceOutput::Policy(actual) = actual else {
            panic!("policy expected")
        };
        for (&a, &b) in actual.logits.iter().zip(&singles[i % 2].logits) {
            if a.is_finite() {
                close(f64::from(a), f64::from(b));
            } else {
                assert_eq!(a, b);
            }
        }
    }
    let seeded = evaluator.evaluate_seeded(&games[0], 17).unwrap();
    let another = evaluator.evaluate_seeded(&games[0], 42).unwrap();
    assert_ne!(seeded.legal, another.legal);
    let repeat = evaluator.evaluate_seeded(&games[0], 17).unwrap();
    assert_eq!(seeded.legal, repeat.legal);
    assert_eq!(seeded.logits, repeat.logits);
    let before = evaluator.stats.encoder_positions;
    let cached = evaluator
        .evaluate_batch(&[(&games[0], false), (&games[1], false)])
        .unwrap();
    assert_eq!(cached.len(), 2);
    assert_eq!(before, evaluator.stats.encoder_positions);
    let terminal = games[0].pass().unwrap().pass().unwrap();
    assert!(
        evaluator
            .evaluate_batch(&[(&terminal, false)])
            .unwrap_err()
            .contains("finished")
    );
    assert!(
        evaluator
            .evaluate_batch(&vec![(&games[0], false); 33])
            .unwrap_err()
            .contains("32")
    );
    assert!(evaluator.evaluate_batch(&[]).unwrap().is_empty());

    evaluator.configure_features(4, 0).unwrap();
    evaluator.stats = Default::default();
    evaluator
        .evaluate_batch(&[(&games[0], false), (&games[0], false)])
        .unwrap();
    assert_eq!(evaluator.stats.encoder_positions, 1);
    assert_eq!(evaluator.stats.coarse_stages.positions, 1);
    assert_eq!(evaluator.stats.cache_entries, 0);
    let variants: Vec<_> = (0..6)
        .map(|i| {
            games[0]
                .play(Spot {
                    x: 400 + 15 * i,
                    y: 600,
                })
                .unwrap()
        })
        .collect();
    let values: Vec<_> = variants
        .iter()
        .map(|g| evaluator.value(g).unwrap())
        .collect();
    evaluator.stats = Default::default();
    let queries: Vec<_> = variants.iter().map(|g| (g, false)).collect();
    let batched_values = evaluator.evaluate_batch(&queries).unwrap();
    assert_eq!(evaluator.stats.encoder_max_batch, 6);
    assert_eq!(evaluator.stats.calls, 0);
    for (actual, expected) in batched_values.iter().zip(values) {
        let InferenceOutput::Value(actual) = actual else {
            panic!("value expected")
        };
        close(actual.value, expected);
    }

    evaluator.configure_features(4, 64).unwrap();
    evaluator.stats = Default::default();
    let (worker, mut client) = BatchingWorker::new(
        evaluator,
        BatchingConfig {
            max_batch: 4,
            max_wait: Duration::from_millis(20),
        },
        4,
    )
    .unwrap();
    let barrier = Barrier::new(3);
    let stats = std::thread::scope(|scope| {
        let worker = scope.spawn(|| worker.run());
        // Partial batch must finish despite there being no other active client.
        close(client.value(&games[0]).unwrap(), singles[0].value);
        // A hit requires no encoder or full policy work.
        close(client.value(&games[0]).unwrap(), singles[0].value);
        assert!(client.value(&terminal).unwrap_err().contains("finished"));
        let actual = client.evaluate_seeded(&games[0], 17).unwrap();
        assert_eq!(actual.legal, seeded.legal);
        for (&a, &b) in actual.logits.iter().zip(&seeded.logits) {
            if a.is_finite() {
                close(f64::from(a), f64::from(b));
            } else {
                assert_eq!(a, b);
            }
        }
        let mut handles = Vec::new();
        for game in [&games[2], &games[3]] {
            let mut client = client.clone();
            let barrier = &barrier;
            handles.push(scope.spawn(move || {
                barrier.wait();
                client.evaluate(game).unwrap()
            }));
        }
        barrier.wait();
        for (handle, expected) in handles.into_iter().zip(&singles[2..4]) {
            let actual = handle.join().unwrap();
            close(actual.value, expected.value);
            assert_eq!(actual.legal, expected.legal);
        }
        drop(client); // Last sender disconnects: worker must exit and join.
        worker.join().unwrap()
    });
    assert_eq!(stats.encoder_max_batch, 2);
    assert_eq!(stats.scorer_max_batch, 2);
    assert_eq!(stats.calls, 3);
    assert_eq!(stats.requests, 6);
    assert_eq!(stats.cache_hits, 2);
}
