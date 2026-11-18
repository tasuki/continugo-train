//! Bounded CPU preparation -> single model-owning execution pipeline.
//! Each game still has at most one outstanding synchronous query.
use super::{
    Evaluation, Evaluator, InferenceOutput, InferenceStats, LeafEstimate, OnnxEvaluator,
    batch::{CoarsePreparer, CoarseRaster, MAX_BATCH},
};
use continugo_rules::Game;
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
pub struct BatchingConfig {
    pub max_batch: usize,
    pub max_wait: Duration,
}
impl BatchingConfig {
    pub fn validate(self) -> Result<(), String> {
        if !(1..=MAX_BATCH).contains(&self.max_batch) {
            return Err("inference batch size must be in 1..=32".into());
        }
        if self.max_wait > Duration::from_secs(1) {
            return Err("inference batch wait must be at most 1000 ms".into());
        }
        Ok(())
    }
}
struct Request {
    game: Game,
    policy: bool,
    seed: u64,
    submitted: Instant,
    reply: SyncSender<Result<InferenceOutput, String>>,
}
#[derive(Clone)]
pub struct BatchingClient {
    sender: SyncSender<Request>,
    model: String,
    name: String,
}
impl BatchingClient {
    fn query(&self, game: &Game, policy: bool, seed: u64) -> Result<InferenceOutput, String> {
        let (reply, receiver) = mpsc::sync_channel(1);
        self.sender
            .send(Request {
                game: game.clone(),
                policy,
                seed,
                submitted: Instant::now(),
                reply,
            })
            .map_err(|_| "inference worker stopped".to_string())?;
        receiver
            .recv()
            .map_err(|_| "inference worker stopped before replying".to_string())?
    }
}
impl Evaluator for BatchingClient {
    fn model_id(&self) -> &str {
        &self.model
    }
    fn network_name(&self) -> &str {
        &self.name
    }
    fn estimate(&mut self, game: &Game) -> Result<LeafEstimate, String> {
        match self.query(game, false, 0)? {
            InferenceOutput::Value(v) => Ok(v),
            _ => Err("wrong inference reply kind".into()),
        }
    }
    fn evaluate(&mut self, game: &Game) -> Result<Evaluation, String> {
        self.evaluate_seeded(game, 0)
    }
    fn evaluate_seeded(&mut self, game: &Game, seed: u64) -> Result<Evaluation, String> {
        match self.query(game, true, seed)? {
            InferenceOutput::Policy(v) => Ok(v),
            _ => Err("wrong inference reply kind".into()),
        }
    }
}

fn reply(stats: &mut InferenceStats, request: Request, result: Result<InferenceOutput, String>) {
    stats.requests += 1;
    stats.request_seconds += request.submitted.elapsed().as_secs_f64();
    // Dropped clients never block either pipeline stage.
    let _ = request.reply.send(result);
}
struct Prepared {
    requests: Vec<Request>,
    rasters: Result<Vec<Option<CoarseRaster>>, String>,
    ready: Instant,
}

fn prepare_requests(
    receiver: Receiver<Request>,
    sender: SyncSender<Prepared>,
    preparer: CoarsePreparer,
    config: BatchingConfig,
) -> InferenceStats {
    let mut stats = InferenceStats::default();
    let admit = |r: Request, pending: &mut Vec<Request>, stats: &mut InferenceStats| {
        if r.game.is_finished() {
            stats.queue_seconds += r.submitted.elapsed().as_secs_f64();
            reply(stats, r, Err("cannot evaluate a finished game".into()));
        } else if !r.policy {
            let cached = preparer.cache.lock().unwrap().get(&r.game);
            if let Some(context) = cached {
                stats.queue_seconds += r.submitted.elapsed().as_secs_f64();
                stats.cache_hits += 1;
                stats.value_calls += 1;
                reply(
                    stats,
                    r,
                    Ok(InferenceOutput::Value(LeafEstimate {
                        value: context.value,
                        score: context.score,
                    })),
                );
            } else {
                pending.push(r);
            }
        } else {
            pending.push(r);
        }
    };
    while let Ok(first) = receiver.recv() {
        let deadline = Instant::now() + config.max_wait;
        let mut pending = Vec::with_capacity(config.max_batch);
        admit(first, &mut pending, &mut stats);
        for _ in 1..config.max_batch {
            if pending.is_empty() {
                break;
            }
            match receiver.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                Ok(r) => admit(r, &mut pending, &mut stats),
                Err(_) => break,
            }
        }
        if pending.is_empty() {
            continue;
        }
        stats.queue_seconds += pending
            .iter()
            .map(|r| r.submitted.elapsed().as_secs_f64())
            .sum::<f64>();
        let tick = Instant::now();
        let games: Vec<_> = pending.iter().map(|r| &r.game).collect();
        let mut rasters = preparer.prepare(&games);
        if let Ok(rasters) = &mut rasters {
            for (_, stages, _) in rasters.iter_mut().flatten() {
                // Count actual CPU work even if execution later finds that a
                // preceding batch has already populated this context's cache.
                stats.coarse_stages.merge(stages);
                *stages = Default::default();
            }
        }
        let seconds = tick.elapsed().as_secs_f64();
        stats.coarse_seconds += seconds;
        stats.feature_seconds += seconds;
        if sender
            .send(Prepared {
                requests: pending,
                rasters,
                ready: Instant::now(),
            })
            .is_err()
        {
            break;
        }
    }
    stats
}

pub struct BatchingWorker {
    evaluator: OnnxEvaluator,
    receiver: Receiver<Request>,
    config: BatchingConfig,
    capacity: usize,
}
impl BatchingWorker {
    pub fn new(
        evaluator: OnnxEvaluator,
        config: BatchingConfig,
        capacity: usize,
    ) -> Result<(Self, BatchingClient), String> {
        config.validate()?;
        if !(1..=64).contains(&capacity) {
            return Err("inference queue capacity must be in 1..=64".into());
        }
        let (sender, receiver) = mpsc::sync_channel(capacity);
        let client = BatchingClient {
            sender,
            model: evaluator.model_id().to_owned(),
            name: evaluator.network_name().to_owned(),
        };
        Ok((
            Self {
                evaluator,
                receiver,
                config,
                capacity,
            },
            client,
        ))
    }
    /// Drain until every game client disconnects. Both stages use bounded
    /// channels; dropped replies and execution errors cannot block shutdown.
    pub fn run(mut self) -> InferenceStats {
        let preparer = self.evaluator.coarse_preparer();
        // Reserve outstanding requests for both stages instead of consuming all
        // game slots in a single batch. The public batch setting remains a cap.
        let config = BatchingConfig {
            max_batch: self.config.max_batch.min(self.capacity.div_ceil(2)),
            ..self.config
        };
        std::thread::scope(|scope| {
            // Drop the receiver before the scope joins on an execution panic,
            // so a producer blocked on the bounded send is released as well.
            let (sender, prepared) = mpsc::sync_channel(1);
            let receiver = self.receiver;
            let producer =
                scope.spawn(move || prepare_requests(receiver, sender, preparer, config));
            while let Ok(batch) = prepared.recv() {
                Self::execute(&mut self.evaluator, batch);
            }
            let cpu = producer.join().expect("coarse preparation worker panicked");
            self.evaluator.stats.requests += cpu.requests;
            self.evaluator.stats.request_seconds += cpu.request_seconds;
            self.evaluator.stats.queue_seconds += cpu.queue_seconds;
            self.evaluator.stats.value_calls += cpu.value_calls;
            self.evaluator.stats.cache_hits += cpu.cache_hits;
            self.evaluator.stats.coarse_seconds += cpu.coarse_seconds;
            self.evaluator.stats.feature_seconds += cpu.feature_seconds;
            self.evaluator.stats.coarse_stages.merge(&cpu.coarse_stages);
        });
        self.evaluator.stats
    }
    fn execute(evaluator: &mut OnnxEvaluator, batch: Prepared) {
        let Prepared {
            requests,
            rasters,
            ready,
        } = batch;
        evaluator.stats.queue_seconds += ready.elapsed().as_secs_f64() * requests.len() as f64;
        let mut rasters = match rasters {
            Ok(r) => r,
            Err(e) => {
                for r in requests {
                    reply(
                        &mut evaluator.stats,
                        r,
                        Err(format!("coarse preparation: {e}")),
                    );
                }
                return;
            }
        };
        let games: Vec<_> = requests.iter().map(|r| &r.game).collect();
        let contexts = match evaluator.contexts_prepared(&games, Some(&mut rasters)) {
            Ok(c) => c,
            Err(e) => {
                let message = format!("batched ONNX contexts: {e}");
                for r in requests {
                    reply(&mut evaluator.stats, r, Err(message.clone()));
                }
                return;
            }
        };
        let mut policies = Vec::new();
        for (request, context) in requests.into_iter().zip(contexts) {
            if request.policy {
                policies.push((request, context));
            } else {
                evaluator.stats.value_calls += 1;
                reply(
                    &mut evaluator.stats,
                    request,
                    Ok(InferenceOutput::Value(LeafEstimate {
                        value: context.value,
                        score: context.score,
                    })),
                );
            }
        }
        let queries: Vec<_> = policies
            .iter()
            .map(|(r, c)| (&r.game, c.clone(), r.seed))
            .collect();
        match evaluator.policies(&queries) {
            Ok(results) => {
                for ((r, _), result) in policies.into_iter().zip(results) {
                    reply(&mut evaluator.stats, r, Ok(InferenceOutput::Policy(result)));
                }
            }
            Err(e) => {
                let message = format!("batched ONNX policies: {e}");
                for (r, _) in policies {
                    reply(&mut evaluator.stats, r, Err(message.clone()));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disconnected_worker_returns_errors_without_hanging() {
        let (sender, receiver) = mpsc::sync_channel(1);
        let mut client = BatchingClient {
            sender,
            model: "test".into(),
            name: "named-test".into(),
        };
        assert_eq!(client.network_name(), "named-test");
        assert_eq!(client.model_id(), "test");
        drop(receiver);
        let game = Game::default();
        assert!(client.value(&game).unwrap_err().contains("stopped"));
        assert!(client.evaluate(&game).unwrap_err().contains("stopped"));
    }
    #[test]
    fn reply_disconnect_is_an_error() {
        let (sender, receiver) = mpsc::sync_channel::<Request>(1);
        let mut client = BatchingClient {
            sender,
            model: "test".into(),
            name: "test".into(),
        };
        std::thread::scope(|scope| {
            scope.spawn(move || {
                drop(receiver.recv().unwrap());
            });
            assert!(
                client
                    .value(&Game::default())
                    .unwrap_err()
                    .contains("before replying")
            );
        });
    }
}
