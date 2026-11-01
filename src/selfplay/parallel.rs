//! Bounded independent neural games. No concurrent simulations within a tree.
use super::{Engine, GameRecord, SelfplayConfig, generate_neural};
use crate::inference::{
    InferenceStats, OnnxEvaluator,
    batching::{BatchingConfig, BatchingWorker},
};
use continugo_rules::Board;
use std::{
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc,
    },
    time::Instant,
};

#[derive(Clone, Copy, Debug)]
pub struct ParallelConfig {
    pub games: u32,
    pub seed: u64,
    pub concurrent_games: usize,
    pub batching: BatchingConfig,
}

impl ParallelConfig {
    pub fn validate(self) -> Result<(), String> {
        self.batching.validate()?;
        if self.games == 0 || !(1..=64).contains(&self.concurrent_games) {
            return Err("positive game count and concurrent games in 1..=64 required".into());
        }
        Ok(())
    }
}

/// Emit on the caller's thread, in completion order. Stable indices determine
/// seeds, board sizes and filenames, independently of worker scheduling. An
/// emitter/game error stops new games and drains in-flight work before joining.
pub fn generate_parallel(
    boards: &[Board],
    config: &SelfplayConfig,
    evaluator: OnnxEvaluator,
    options: ParallelConfig,
    mut emit: impl FnMut(u32, GameRecord, f64) -> Result<(), String>,
) -> Result<InferenceStats, String> {
    options.validate()?;
    config.validate()?;
    if boards.is_empty() || !matches!(config.engine, Engine::Neural) {
        return Err("parallel self-play requires boards and the neural engine".into());
    }
    let concurrency = options.concurrent_games.min(options.games as usize);
    let mut batching = options.batching;
    batching.max_batch = batching.max_batch.min(concurrency);
    // A single active game cannot supply a second outstanding request.
    if concurrency == 1 {
        batching.max_wait = std::time::Duration::ZERO;
    }
    let (worker, client) = BatchingWorker::new(evaluator, batching, concurrency)?;
    let (sender, receiver) = mpsc::sync_channel(concurrency);
    let next = AtomicUsize::new(0);
    let cancelled = AtomicBool::new(false);
    std::thread::scope(|scope| {
        let inference = scope.spawn(move || worker.run());
        let mut games = Vec::new();
        for _ in 0..concurrency {
            let mut client = client.clone();
            let sender = sender.clone();
            let next = &next;
            let cancelled = &cancelled;
            games.push(scope.spawn(move || {
                while !cancelled.load(Ordering::Relaxed) {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= options.games as usize {
                        break;
                    }
                    let started = Instant::now();
                    let result = generate_neural(
                        boards[index % boards.len()],
                        config,
                        options.seed.wrapping_add(index as u64),
                        &mut client,
                    );
                    if result.is_err() {
                        cancelled.store(true, Ordering::Relaxed);
                    }
                    if sender
                        .send((index as u32, result, started.elapsed().as_secs_f64()))
                        .is_err()
                    {
                        break;
                    }
                }
            }));
        }
        drop(client);
        drop(sender);
        let mut error = None;
        for (index, result, seconds) in receiver {
            let result = match result {
                Ok(record) if error.is_none() => emit(index, record, seconds),
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
                error.get_or_insert("self-play worker panicked".into());
            }
        }
        let stats = inference
            .join()
            .map_err(|_| "inference worker panicked".to_string())?;
        match error {
            Some(error) => Err(error),
            None => Ok(stats),
        }
    })
}
