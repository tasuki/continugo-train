//! Resolution-bucketed encoder/scorer execution with bounded streaming move rows.
use super::{Evaluation, OnnxEvaluator, cache::Context};
use crate::features::{COARSE_PLANES, FeatureExtractor, MOVE_FEATURES, resolution};
use continugo_rules::{COORDINATE_ACTIONS, Game};
use ort::value::Tensor;
use rayon::prelude::*;
use std::{collections::BTreeMap, sync::Arc, time::Instant};

pub const MAX_BATCH: usize = 32;

#[derive(Debug)]
pub enum InferenceOutput {
    Value(f64),
    Policy(Evaluation),
}

type Error = Box<dyn std::error::Error>;

impl OnnxEvaluator {
    /// Each tuple requests either a full-domain policy (true) or only a value.
    /// Mixed resolutions are bucketed, not padded. Outputs retain request order.
    pub fn evaluate_batch(
        &mut self,
        queries: &[(&Game, bool)],
    ) -> Result<Vec<InferenceOutput>, String> {
        let games: Vec<_> = queries.iter().map(|q| q.0).collect();
        let contexts = self
            .contexts(&games)
            .map_err(|e| format!("ONNX contexts: {e}"))?;
        let policies: Vec<_> = queries
            .iter()
            .zip(&contexts)
            .filter(|(q, _)| q.1)
            .map(|(q, c)| (q.0, Arc::clone(c)))
            .collect();
        let mut evaluations = self
            .policies(&policies)
            .map_err(|e| format!("ONNX policies: {e}"))?
            .into_iter();
        Ok(queries
            .iter()
            .zip(contexts)
            .map(|(q, c)| {
                if q.1 {
                    InferenceOutput::Policy(evaluations.next().unwrap())
                } else {
                    self.stats.value_calls += 1;
                    InferenceOutput::Value(c.value)
                }
            })
            .collect())
    }

    pub(super) fn cached_context(&mut self, game: &Game) -> Option<Arc<Context>> {
        let context = self.cache.get(game)?;
        self.stats.cache_hits += 1;
        self.cache_stats();
        Some(context)
    }

    fn cache_stats(&mut self) {
        self.stats.cache_entries = self.cache.len();
        self.stats.cache_bytes = self.cache.bytes;
    }

    pub(super) fn contexts(&mut self, games: &[&Game]) -> Result<Vec<Arc<Context>>, Error> {
        if games.len() > MAX_BATCH {
            return Err("inference batch exceeds 32 positions".into());
        }
        if games.iter().any(|g| g.is_finished()) {
            return Err("cannot evaluate a finished game".into());
        }
        let mut result = vec![None; games.len()];
        let mut pending: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let mut duplicates = Vec::new();
        for (i, game) in games.iter().enumerate() {
            if let Some(context) = self.cached_context(game) {
                result[i] = Some(context);
                continue;
            }
            self.stats.cache_misses += 1;
            // Identical openings in independent games need only one execution,
            // even when caching is disabled or the payload exceeds its budget.
            if let Some(j) = games[..i].iter().position(|other| *other == *game) {
                duplicates.push((i, j));
            } else {
                let size = resolution(
                    game.board().diameter(),
                    self.metadata.feature_spec.samples_per_diameter,
                )?;
                pending.entry(size).or_default().push(i);
            }
        }
        for (size, indices) in pending {
            let tick = Instant::now();
            let density = self.metadata.feature_spec.samples_per_diameter;
            let extract = |&i: &usize| FeatureExtractor::new(games[i], density).map(|e| e.coarse);
            let rasters: Result<Vec<_>, _> = match &self.feature_pool {
                Some(pool) => pool.install(|| indices.par_iter().map(extract).collect()),
                None => indices.iter().map(extract).collect(),
            };
            let coarse: Vec<f32> = rasters?.into_iter().flatten().collect();
            let seconds = tick.elapsed().as_secs_f64();
            self.stats.coarse_seconds += seconds;
            self.stats.feature_seconds += seconds;
            let tick = Instant::now();
            let batch = indices.len();
            let encoded = self.encoder.run(ort::inputs![
                "coarse" => Tensor::from_array(([batch, COARSE_PLANES.len(), size, size], coarse))?
            ]?)?;
            let (shape, spatial) = encoded["spatial"].try_extract_raw_tensor::<f32>()?;
            let (_, pass) = encoded["pass_logit"].try_extract_raw_tensor::<f32>()?;
            let (_, value) = encoded["value"].try_extract_raw_tensor::<f32>()?;
            let stride = self.metadata.config.policy_hidden * size * size;
            if shape
                != [
                    batch as i64,
                    self.metadata.config.policy_hidden as i64,
                    size as i64,
                    size as i64,
                ]
                || pass.len() != batch
                || value.len() != batch
                || spatial.iter().chain(pass).any(|v| !v.is_finite())
                || value
                    .iter()
                    .any(|v| !v.is_finite() || !(-1.0..=1.0).contains(v))
            {
                return Err("invalid encoder outputs".into());
            }
            for (row, i) in indices.into_iter().enumerate() {
                let context = Arc::new(Context {
                    resolution: size,
                    spatial: spatial[row * stride..(row + 1) * stride].to_vec(),
                    pass: pass[row],
                    value: f64::from(value[row]),
                });
                self.stats.cache_evictions +=
                    self.cache.insert(games[i], Arc::clone(&context)) as u64;
                result[i] = Some(context);
            }
            self.stats.encoder_seconds += tick.elapsed().as_secs_f64();
            self.stats.encoder_positions += batch as u64;
            self.stats.encoder_batches += 1;
            self.stats.encoder_max_batch = self.stats.encoder_max_batch.max(batch);
        }
        for (i, j) in duplicates {
            result[i] = result[j].clone();
        }
        self.cache_stats();
        Ok(result.into_iter().map(Option::unwrap).collect())
    }

    pub(super) fn policies(
        &mut self,
        queries: &[(&Game, Arc<Context>)],
    ) -> Result<Vec<Evaluation>, Error> {
        if queries.len() > MAX_BATCH {
            return Err("inference batch exceeds 32 positions".into());
        }
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let mut result: Vec<Option<Evaluation>> = (0..queries.len()).map(|_| None).collect();
        for (i, (_, context)) in queries.iter().enumerate() {
            groups.entry(context.resolution).or_default().push(i);
        }
        for (size, indices) in groups {
            let batch = indices.len();
            let tick = Instant::now();
            let extractors: Vec<_> = indices
                .iter()
                .map(|&i| FeatureExtractor::for_moves(queries[i].0))
                .collect::<Result<_, _>>()?;
            let seconds = tick.elapsed().as_secs_f64();
            self.stats.feature_seconds += seconds;
            self.stats.move_seconds += seconds;
            let spatial: Vec<f32> = indices
                .iter()
                .flat_map(|&i| queries[i].1.spatial.iter().copied())
                .collect();
            for &i in &indices {
                result[i] = Some(Evaluation {
                    value: queries[i].1.value,
                    logits: Vec::with_capacity(COORDINATE_ACTIONS as usize + 1),
                    legal: Vec::with_capacity(COORDINATE_ACTIONS as usize),
                });
            }
            // At most 262,144 exact rows across a scorer call, not B million-row
            // matrices. Persistent outputs cost about 5 MB per policy request.
            let chunk = self
                .metadata
                .config
                .chunk_size
                .min(65_536)
                .min(262_144 / batch);
            for start in (0..COORDINATE_ACTIONS as usize).step_by(chunk) {
                let end = (start + chunk).min(COORDINATE_ACTIONS as usize);
                let tick = Instant::now();
                let actions = &self.actions[start..end];
                let moves: Result<Vec<_>, _> = match &self.feature_pool {
                    Some(pool) => pool.install(|| {
                        extractors
                            .par_iter()
                            .map(|e| e.moves_parallel(actions, pool))
                            .collect()
                    }),
                    None => extractors.iter().map(|e| e.moves(actions)).collect(),
                };
                let moves = moves?;
                let exact: Vec<f32> = moves.iter().flat_map(|m| m.exact.iter().copied()).collect();
                let seconds = tick.elapsed().as_secs_f64();
                self.stats.feature_seconds += seconds;
                self.stats.move_seconds += seconds;
                let tick = Instant::now();
                let ids: Vec<i64> = (start as i64..end as i64).collect();
                let output = self.scorer.run(ort::inputs![
                    "spatial" => Tensor::from_array(([batch, self.metadata.config.policy_hidden, size, size], spatial.as_slice()))?,
                    "exact" => Tensor::from_array(([batch, end-start, MOVE_FEATURES.len()], exact))?,
                    "actions" => Tensor::from_array(([end-start], ids))?
                ]?)?;
                let (shape, scores) = output["logits"].try_extract_raw_tensor::<f32>()?;
                if shape != [batch as i64, (end - start) as i64]
                    || scores.iter().any(|v| !v.is_finite())
                {
                    return Err("invalid scorer outputs".into());
                }
                for ((&i, moves), scores) in indices
                    .iter()
                    .zip(moves)
                    .zip(scores.chunks_exact(end - start))
                {
                    let evaluation = result[i].as_mut().unwrap();
                    evaluation
                        .logits
                        .extend(scores.iter().zip(&moves.legal).map(|(&score, &legal)| {
                            if legal == 0 { f32::NEG_INFINITY } else { score }
                        }));
                    evaluation.legal.extend(moves.legal);
                }
                self.stats.scorer_seconds += tick.elapsed().as_secs_f64();
                self.stats.scorer_batches += 1;
                self.stats.scorer_max_batch = self.stats.scorer_max_batch.max(batch);
            }
            for &i in &indices {
                result[i].as_mut().unwrap().logits.push(queries[i].1.pass);
            }
            self.stats.calls += batch as u64;
            self.stats.scorer_positions += batch as u64;
        }
        Ok(result.into_iter().map(Option::unwrap).collect())
    }
}
