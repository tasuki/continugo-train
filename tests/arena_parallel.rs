mod support;
use continugo_rules::Board;
use continugo_train::{
    inference::batching::BatchingConfig,
    selfplay::{Engine, SelfplayConfig, arena::generate_arena, parallel::ParallelConfig},
};
use std::{collections::BTreeMap, time::Duration};

fn config() -> SelfplayConfig {
    SelfplayConfig {
        engine: Engine::Mcts,
        komi: 7.0,
        komi_stddev: 0.0,
        max_moves: 2,
        search: continugo_train::search::SearchConfig {
            simulations: 2,
            ..support::search_config()
        },
        policy_search: None,
    }
}

fn options(concurrency: usize) -> ParallelConfig {
    ParallelConfig {
        games: 5,
        seed: 71,
        concurrent_games: concurrency,
        batching: BatchingConfig {
            max_batch: 4,
            max_wait: Duration::from_millis(2),
        },
    }
}

#[test]
fn paired_ordinals_survive_parallel_completion_and_partial_resume() {
    let boards = [Board::new(9).unwrap(), Board::new(13).unwrap()];
    let config = config();
    let run = |concurrency| {
        let mut records = BTreeMap::new();
        let stats = generate_arena(
            &boards,
            &config,
            None,
            None,
            3,
            options(concurrency),
            |ordinal, record, _| {
                assert!(records.insert(ordinal, record.sgf).is_none());
                Ok(())
            },
        )
        .unwrap();
        assert!(stats.is_empty());
        records
    };
    let serial = run(1);
    assert_eq!(
        serial.keys().copied().collect::<Vec<_>>(),
        vec![3, 4, 5, 6, 7]
    );
    assert_eq!(serial, run(4));
    for (ordinal, sgf) in serial {
        let expected = continugo_train::selfplay::generate_unguided_reference_match(
            boards[(ordinal / 2) % boards.len()],
            &config,
            71 + (ordinal / 2) as u64,
            ordinal % 2 == 0,
        )
        .unwrap();
        assert_eq!(sgf, expected.sgf);
    }
}

#[test]
fn emitter_error_cancels_and_drains_active_games() {
    let boards = [Board::new(9).unwrap()];
    let config = config();
    let mut emitted = 0;
    let result = generate_arena(&boards, &config, None, None, 0, options(4), |_, _, _| {
        emitted += 1;
        Err("output failed".into())
    });
    assert_eq!(result.unwrap_err(), "output failed");
    assert_eq!(emitted, 1);
}

#[test]
fn invalid_limits_fail_before_starting_workers() {
    let config = config();
    let boards = [Board::new(9).unwrap()];
    for concurrency in [0, 65] {
        assert!(
            generate_arena(
                &boards,
                &config,
                None,
                None,
                0,
                options(concurrency),
                |_, _, _| Ok(())
            )
            .is_err()
        );
    }
    assert!(
        generate_arena(
            &boards,
            &config,
            None,
            None,
            usize::MAX,
            options(4),
            |_, _, _| Ok(())
        )
        .is_err()
    );
    assert!(
        generate_arena(
            &boards,
            &config,
            None,
            None,
            0,
            ParallelConfig {
                seed: u64::MAX,
                ..options(4)
            },
            |_, _, _| Ok(())
        )
        .is_err()
    );
}
