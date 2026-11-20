//! Independent paired-color games sharing one bounded inference pipeline per model.
use super::{
    GameRecord, SelfplayConfig, generate_match, generate_reference_match,
    generate_unguided_reference_match, parallel::ParallelConfig,
};
use crate::inference::{InferenceStats, OnnxEvaluator, batching::BatchingWorker};
use continugo_rules::Board;
use std::{
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    time::Instant,
};

/// Emit on the caller's thread in completion order. Ordinals, not scheduling,
/// determine size, pair seed and candidate color. Errors cancel new work while
/// draining active games; dropping every client shuts down both model workers.
pub fn generate_arena(
    boards: &[Board],
    config: &SelfplayConfig,
    candidate: Option<OnnxEvaluator>,
    incumbent: Option<OnnxEvaluator>,
    game_offset: usize,
    options: ParallelConfig,
    mut emit: impl FnMut(usize, GameRecord, f64) -> Result<(), String>,
) -> Result<Vec<InferenceStats>, String> {
    options.validate()?;
    config.validate()?;
    if boards.is_empty() || incumbent.is_some() && candidate.is_none() {
        return Err("arena requires boards and a candidate for neural opponents".into());
    }
    let end = game_offset
        .checked_add(options.games as usize)
        .ok_or("game ordinal overflow")?;
    options
        .seed
        .checked_add(((end - 1) / 2) as u64)
        .ok_or("pair seed overflow")?;
    let concurrency = options.concurrent_games.min(options.games as usize);
    let mut batching = options.batching;
    batching.max_batch = batching.max_batch.min(concurrency);
    if concurrency == 1 {
        batching.max_wait = std::time::Duration::ZERO;
    }
    let candidate = candidate
        .map(|e| BatchingWorker::new(e, batching, concurrency))
        .transpose()?;
    let incumbent = incumbent
        .map(|e| BatchingWorker::new(e, batching, concurrency))
        .transpose()?;
    let next = AtomicUsize::new(game_offset);
    let cancelled = AtomicBool::new(false);
    let (sender, receiver) = mpsc::sync_channel(concurrency);
    std::thread::scope(|scope| {
        let (candidate_worker, candidate_client) = match candidate {
            Some((worker, client)) => (Some(scope.spawn(move || worker.run())), Some(client)),
            None => (None, None),
        };
        let (incumbent_worker, incumbent_client) = match incumbent {
            Some((worker, client)) => (Some(scope.spawn(move || worker.run())), Some(client)),
            None => (None, None),
        };
        let mut games = Vec::new();
        for _ in 0..concurrency {
            let mut candidate = candidate_client.clone();
            let mut incumbent = incumbent_client.clone();
            let sender = sender.clone();
            let next = &next;
            let cancelled = &cancelled;
            games.push(scope.spawn(move || {
                while !cancelled.load(Ordering::Relaxed) {
                    let ordinal = next.fetch_add(1, Ordering::Relaxed);
                    if ordinal >= end {
                        break;
                    }
                    let board = boards[(ordinal / 2) % boards.len()];
                    let seed = options.seed + (ordinal / 2) as u64;
                    let candidate_black = ordinal % 2 == 0;
                    let started = Instant::now();
                    let result = match (&mut candidate, &mut incumbent) {
                        (Some(candidate), Some(incumbent)) if candidate_black => {
                            generate_match(board, config, seed, candidate, incumbent)
                        }
                        (Some(candidate), Some(incumbent)) => {
                            generate_match(board, config, seed, incumbent, candidate)
                        }
                        (Some(candidate), None) => generate_reference_match(
                            board,
                            config,
                            seed,
                            candidate,
                            candidate_black,
                        ),
                        (None, _) => {
                            generate_unguided_reference_match(board, config, seed, candidate_black)
                        }
                    };
                    if result.is_err() {
                        cancelled.store(true, Ordering::Relaxed);
                    }
                    if sender
                        .send((ordinal, result, started.elapsed().as_secs_f64()))
                        .is_err()
                    {
                        break;
                    }
                }
            }));
        }
        drop(candidate_client);
        drop(incumbent_client);
        drop(sender);
        let mut error = None;
        for (ordinal, result, seconds) in receiver {
            let result = match result {
                Ok(record) if error.is_none() => emit(ordinal, record, seconds),
                Ok(_) => Ok(()),
                Err(message) => Err(message),
            };
            if let Err(message) = result {
                cancelled.store(true, Ordering::Relaxed);
                error.get_or_insert(message);
            }
        }
        for game in games {
            if game.join().is_err() {
                error.get_or_insert("arena game worker panicked".into());
            }
        }
        let mut stats = Vec::new();
        for worker in [candidate_worker, incumbent_worker].into_iter().flatten() {
            match worker.join() {
                Ok(value) => stats.push(value),
                Err(_) => {
                    error.get_or_insert("arena inference worker panicked".into());
                }
            }
        }
        match error {
            Some(error) => Err(error),
            None => Ok(stats),
        }
    })
}
