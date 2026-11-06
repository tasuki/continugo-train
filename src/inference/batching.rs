//! One model-owning worker; synchronous game clients have one outstanding query.
//! Queue capacity, active games and batch size are independently bounded.
use super::{
    Evaluation, Evaluator, InferenceOutput, InferenceStats, LeafEstimate, OnnxEvaluator,
    batch::MAX_BATCH,
};
use continugo_rules::Game;
use std::{
    sync::mpsc::{self, Receiver, SyncSender},
    time::{Duration, Instant},
};

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
            InferenceOutput::Value(value) => Ok(value),
            _ => Err("wrong inference reply kind".into()),
        }
    }
    fn evaluate(&mut self, game: &Game) -> Result<Evaluation, String> {
        self.evaluate_seeded(game, 0)
    }
    fn evaluate_seeded(&mut self, game: &Game, seed: u64) -> Result<Evaluation, String> {
        match self.query(game, true, seed)? {
            InferenceOutput::Policy(evaluation) => Ok(evaluation),
            _ => Err("wrong inference reply kind".into()),
        }
    }
}

pub struct BatchingWorker {
    evaluator: OnnxEvaluator,
    receiver: Receiver<Request>,
    config: BatchingConfig,
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
            },
            client,
        ))
    }

    /// Drain until every game client is dropped. Dropped reply receivers never
    /// block shutdown; execution errors are returned to every affected caller.
    pub fn run(mut self) -> InferenceStats {
        while let Ok(first) = self.receiver.recv() {
            let deadline = Instant::now() + self.config.max_wait;
            let mut pending = Vec::with_capacity(self.config.max_batch);
            self.admit(first, &mut pending);
            // Bound collection by total admissions as well as by time: a stream
            // of cached value requests must not starve an uncached query.
            for _ in 1..self.config.max_batch {
                // A cache-only request should not incur a collection delay.
                if pending.is_empty() {
                    break;
                }
                match self
                    .receiver
                    .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                {
                    Ok(request) => self.admit(request, &mut pending),
                    Err(_) => break,
                }
            }
            self.execute(pending);
        }
        self.evaluator.stats
    }

    fn admit(&mut self, request: Request, pending: &mut Vec<Request>) {
        if request.game.is_finished() {
            self.evaluator.stats.queue_seconds += request.submitted.elapsed().as_secs_f64();
            self.reply(request, Err("cannot evaluate a finished game".into()));
        } else if !request.policy {
            if let Some(context) = self.evaluator.cached_context(&request.game) {
                self.evaluator.stats.queue_seconds += request.submitted.elapsed().as_secs_f64();
                self.evaluator.stats.value_calls += 1;
                self.reply(
                    request,
                    Ok(InferenceOutput::Value(LeafEstimate {
                        value: context.value,
                        score: context.score,
                    })),
                );
            } else {
                pending.push(request);
            }
        } else {
            pending.push(request);
        }
    }

    fn reply(&mut self, request: Request, result: Result<InferenceOutput, String>) {
        self.evaluator.stats.requests += 1;
        self.evaluator.stats.request_seconds += request.submitted.elapsed().as_secs_f64();
        let _ = request.reply.send(result);
    }

    fn execute(&mut self, requests: Vec<Request>) {
        if requests.is_empty() {
            return;
        }
        self.evaluator.stats.queue_seconds += requests
            .iter()
            .map(|r| r.submitted.elapsed().as_secs_f64())
            .sum::<f64>();
        let games: Vec<_> = requests.iter().map(|r| &r.game).collect();
        let contexts = match self.evaluator.contexts(&games) {
            Ok(contexts) => contexts,
            Err(error) => {
                let message = format!("batched ONNX contexts: {error}");
                for request in requests {
                    self.reply(request, Err(message.clone()));
                }
                return;
            }
        };
        let mut policies = Vec::new();
        for (request, context) in requests.into_iter().zip(contexts) {
            if request.policy {
                policies.push((request, context));
            } else {
                // Do not make value leaves wait for dense feature/scorer work.
                self.evaluator.stats.value_calls += 1;
                self.reply(
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
        match self.evaluator.policies(&queries) {
            Ok(results) => {
                for ((request, _), result) in policies.into_iter().zip(results) {
                    self.reply(request, Ok(InferenceOutput::Policy(result)));
                }
            }
            Err(error) => {
                let message = format!("batched ONNX policies: {error}");
                for (request, _) in policies {
                    self.reply(request, Err(message.clone()));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use continugo_rules::Board;

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
        let game = Game::new(Board::new(9).unwrap());
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
                    .value(&Game::new(Board::new(9).unwrap()))
                    .unwrap_err()
                    .contains("before replying")
            );
        });
    }
}
