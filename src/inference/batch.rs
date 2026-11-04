//! Resolution-bucketed encoding and selective, complete-cell exact scoring.
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
    /// Each tuple requests either a selective cell policy (true) or only a value.
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
            .map(|(q, c)| (q.0, Arc::clone(c), 0))
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
            let profile = self.profile_coarse;
            let extract = |&i: &usize| {
                if profile {
                    FeatureExtractor::profiled(games[i], density)
                        .map(|(e, stages)| (e.coarse, stages))
                } else {
                    FeatureExtractor::new(games[i], density).map(|e| (e.coarse, Default::default()))
                }
            };
            let rasters: Result<Vec<_>, _> = match &self.feature_pool {
                Some(pool) => pool.install(|| indices.par_iter().map(extract).collect()),
                None => indices.iter().map(extract).collect(),
            };
            let mut coarse = Vec::with_capacity(indices.len() * COARSE_PLANES.len() * size * size);
            for (raster, stages) in rasters? {
                coarse.extend(raster);
                self.stats.coarse_stages.merge(&stages);
            }
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
            let (cell_shape, cells) = encoded["cells"].try_extract_raw_tensor::<f32>()?;
            let stride = self.metadata.config.policy_hidden * size * size;
            if shape
                != [
                    batch as i64,
                    self.metadata.config.policy_hidden as i64,
                    size as i64,
                    size as i64,
                ]
                || cell_shape != [batch as i64, (size * size) as i64]
                || cells.iter().any(|v| !v.is_finite())
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
                    cells: cells[row * size * size..(row + 1) * size * size].to_vec(),
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
        queries: &[(&Game, Arc<Context>, u64)],
    ) -> Result<Vec<Evaluation>, Error> {
        if queries.len() > MAX_BATCH {
            return Err("inference batch exceeds 32 positions".into());
        }
        let mut groups: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let mut result: Vec<Option<Evaluation>> = (0..queries.len()).map(|_| None).collect();
        for (i, (_, context, _)) in queries.iter().enumerate() {
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
            let actions: Vec<_> = indices
                .iter()
                .map(|&i| {
                    let selected = super::proposals::select(
                        queries[i].0,
                        &queries[i].1.cells,
                        self.metadata.config.top_cells,
                        self.metadata.config.exploration_cells,
                        queries[i].2,
                    );
                    self.stats.selected_cells += selected.len() as u64;
                    crate::cells::actions(&selected, size)
                })
                .collect();
            let count = actions.iter().map(Vec::len).max().unwrap_or(0);
            let mut raw: Vec<Vec<f32>> = actions
                .iter()
                .map(|a| vec![f32::NEG_INFINITY; a.len()])
                .collect();
            for (&i, actions) in indices.iter().zip(&actions) {
                self.stats.exact_coordinates += actions.len() as u64;
                result[i] = Some(Evaluation {
                    value: queries[i].1.value,
                    // Dense IDs are only the search interface, not dense rules work.
                    logits: vec![f32::NEG_INFINITY; COORDINATE_ACTIONS as usize + 1],
                    legal: vec![0; COORDINATE_ACTIONS as usize],
                });
            }
            let chunk = self
                .metadata
                .config
                .chunk_size
                .min(65_536)
                .min(262_144 / batch);
            for start in (0..count).step_by(chunk) {
                let end = (start + chunk).min(count);
                let width = end - start;
                let tick = Instant::now();
                let extract = |row: usize| {
                    let subset =
                        &actions[row][start.min(actions[row].len())..end.min(actions[row].len())];
                    match &self.feature_pool {
                        Some(pool) => extractors[row].moves_parallel(subset, pool),
                        None => extractors[row].moves(subset),
                    }
                };
                let moves: Vec<_> = match &self.feature_pool {
                    Some(pool) => pool.install(|| {
                        (0..batch)
                            .into_par_iter()
                            .map(extract)
                            .collect::<Result<Vec<_>, _>>()
                    }),
                    None => (0..batch).map(extract).collect(),
                }?;
                let mut exact = vec![0.0; batch * width * MOVE_FEATURES.len()];
                let mut ids = vec![0i64; batch * width];
                for (row, moves) in moves.iter().enumerate() {
                    let offset = row * width * MOVE_FEATURES.len();
                    exact[offset..offset + moves.exact.len()].copy_from_slice(&moves.exact);
                    for j in 0..moves.legal.len() {
                        ids[row * width + j] = actions[row][start + j].index() as i64;
                    }
                }
                let seconds = tick.elapsed().as_secs_f64();
                self.stats.feature_seconds += seconds;
                self.stats.move_seconds += seconds;
                let tick = Instant::now();
                let output = self.scorer.run(ort::inputs![
                    "spatial" => Tensor::from_array(([batch, self.metadata.config.policy_hidden, size, size], spatial.as_slice()))?,
                    "exact" => Tensor::from_array(([batch, width, MOVE_FEATURES.len()], exact))?,
                    "actions" => Tensor::from_array(([batch, width], ids))?
                ]?)?;
                let (shape, scores) = output["logits"].try_extract_raw_tensor::<f32>()?;
                if shape != [batch as i64, width as i64] || scores.iter().any(|v| !v.is_finite()) {
                    return Err("invalid scorer outputs".into());
                }
                for (row, (&i, moves)) in indices.iter().zip(&moves).enumerate() {
                    for j in 0..moves.legal.len() {
                        if moves.legal[j] != 0 {
                            raw[row][start + j] = scores[row * width + j];
                            result[i].as_mut().unwrap().legal
                                [actions[row][start + j].index() as usize] = 1;
                        }
                    }
                }
                self.stats.scorer_seconds += tick.elapsed().as_secs_f64();
                self.stats.scorer_batches += 1;
                self.stats.scorer_max_batch = self.stats.scorer_max_batch.max(batch);
            }
            for (row, &i) in indices.iter().enumerate() {
                let context = &queries[i].1;
                let mut maxima = vec![f64::NEG_INFINITY; size * size];
                for (action, &score) in actions[row].iter().zip(&raw[row]) {
                    let cell = crate::cells::cell(action.index(), size);
                    maxima[cell] = maxima[cell].max(f64::from(score));
                }
                let mut sums = vec![0.0f64; size * size];
                for (action, &score) in actions[row].iter().zip(&raw[row]) {
                    if score.is_finite() {
                        let cell = crate::cells::cell(action.index(), size);
                        sums[cell] += (f64::from(score) - maxima[cell]).exp();
                    }
                }
                let coarse_max = context.cells.iter().copied().fold(context.pass, f32::max) as f64;
                let coarse_logsum = (context
                    .cells
                    .iter()
                    .map(|&x| (f64::from(x) - coarse_max).exp())
                    .sum::<f64>()
                    + (f64::from(context.pass) - coarse_max).exp())
                .ln();
                let evaluation = result[i].as_mut().unwrap();
                for (action, &score) in actions[row].iter().zip(&raw[row]) {
                    if score.is_finite() {
                        let cell = crate::cells::cell(action.index(), size);
                        let coarse_logp =
                            f64::from(context.cells[cell]) - coarse_max - coarse_logsum;
                        let local_logp = f64::from(score) - maxima[cell] - sums[cell].ln();
                        evaluation.logits[action.index() as usize] =
                            (coarse_logp + local_logp) as f32;
                    }
                }
                evaluation.logits[COORDINATE_ACTIONS as usize] =
                    (f64::from(context.pass) - coarse_max - coarse_logsum) as f32;
            }
            self.stats.calls += batch as u64;
            self.stats.scorer_positions += batch as u64;
        }
        Ok(result.into_iter().map(Option::unwrap).collect())
    }
}
