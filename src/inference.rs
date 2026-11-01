//! Native CPU ONNX inference. No Python process participates in self-play.
//! Full-domain logits are streamed in chunks; only sampled candidates survive
//! in the tree. The library is loaded only for neural commands, not baselines.

use crate::{
    action::Action,
    features::{COARSE_PLANES, FeatureExtractor, MOVE_FEATURES, SCHEMA_VERSION},
};
use continugo_rules::{COORDINATE_ACTIONS, Game};
use ort::{session::Session, value::TensorRef};
use serde::Deserialize;
use std::{path::Path, time::Instant};

pub struct Evaluation {
    pub value: f64,
    /// Raw logits in shared action-ID order, including pass at 1,000,000.
    pub logits: Vec<f32>,
    pub legal: Vec<u8>,
}

pub trait Evaluator {
    fn evaluate(&mut self, game: &Game) -> Result<Evaluation, String>;
    fn model_id(&self) -> &str;
    fn value(&mut self, game: &Game) -> Result<f64, String> {
        Ok(self.evaluate(game)?.value)
    }
}

#[derive(Deserialize)]
struct FeatureSpec {
    schema: u32,
    coarse: Vec<String>,
    moves: Vec<String>,
    samples_per_diameter: usize,
}

#[derive(Deserialize)]
struct Config {
    chunk_size: usize,
    policy_hidden: usize,
}

#[derive(Deserialize)]
struct Metadata {
    checkpoint_sha256: String,
    config: Config,
    feature_spec: FeatureSpec,
}

#[derive(Debug, Default)]
pub struct InferenceStats {
    pub calls: u64,
    pub value_calls: u64,
    pub feature_seconds: f64,
    pub encoder_seconds: f64,
    pub scorer_seconds: f64,
}

pub struct OnnxEvaluator {
    encoder: Session,
    scorer: Session,
    metadata: Metadata,
    actions: Vec<Action>,
    pub stats: InferenceStats,
}

impl OnnxEvaluator {
    pub fn load(directory: &Path, library: Option<&Path>, threads: usize) -> Result<Self, String> {
        Self::load_inner(directory, library, threads).map_err(|error| format!("ONNX load: {error}"))
    }

    fn load_inner(
        directory: &Path,
        library: Option<&Path>,
        threads: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if threads == 0 {
            return Err("ONNX thread count must be positive".into());
        }
        // Resolve explicitly before calling ort: its lazy loader panics on a missing library.
        let library = library.map(Path::to_path_buf)
            .or_else(|| std::env::var_os("ORT_DYLIB_PATH").map(Into::into))
            .ok_or("set ORT_DYLIB_PATH or --ort-library to an ONNX Runtime 1.22 shared library (see docs/inference.md)")?;
        if !library.is_file() {
            return Err(format!("runtime library does not exist: {}", library.display()).into());
        }
        ort::init_from(library.to_string_lossy())
            .with_telemetry(false)
            .commit()?;
        let load = |name| -> ort::Result<Session> {
            Session::builder()?
                .with_intra_threads(threads)?
                .with_inter_threads(1)?
                .with_config_entry("session.intra_op.allow_spinning", "0")?
                .with_config_entry("session.inter_op.allow_spinning", "0")?
                .commit_from_file(directory.join(name))
        };
        let encoder = load("encoder.onnx")?;
        let scorer = load("policy.onnx")?;
        let encoder_metadata = encoder
            .metadata()?
            .custom("continugo")?
            .ok_or("missing encoder feature metadata")?;
        let scorer_metadata = scorer
            .metadata()?
            .custom("continugo")?
            .ok_or("missing scorer feature metadata")?;
        if encoder_metadata != scorer_metadata {
            return Err("encoder/scorer metadata do not match (mixed model files)".into());
        }
        let metadata: Metadata = serde_json::from_str(&encoder_metadata)?;
        let spec = &metadata.feature_spec;
        if spec.schema != SCHEMA_VERSION
            || spec.coarse != COARSE_PLANES
            || spec.moves != MOVE_FEATURES
            || spec.samples_per_diameter == 0
            || metadata.config.chunk_size == 0
            || metadata.config.policy_hidden == 0
            || metadata.checkpoint_sha256.len() != 64
            || !metadata
                .checkpoint_sha256
                .bytes()
                .all(|c| c.is_ascii_hexdigit())
        {
            return Err("incompatible feature/architecture metadata".into());
        }
        Ok(Self {
            encoder,
            scorer,
            metadata,
            actions: (0..COORDINATE_ACTIONS)
                .map(|i| Action::from_index(i).unwrap())
                .collect(),
            stats: InferenceStats::default(),
        })
    }

    fn value_inner(&mut self, game: &Game) -> Result<f64, Box<dyn std::error::Error>> {
        let tick = Instant::now();
        let extractor =
            FeatureExtractor::new(game, self.metadata.feature_spec.samples_per_diameter)?;
        let resolution = extractor.resolution;
        self.stats.feature_seconds += tick.elapsed().as_secs_f64();
        let tick = Instant::now();
        let encoded = self
            .encoder
            .run(ort::inputs!["coarse" => TensorRef::from_array_view(
            ([1, COARSE_PLANES.len(), resolution, resolution], extractor.coarse.as_slice()))?])?;
        let (_, value) = encoded["value"].try_extract_tensor::<f32>()?;
        if value.len() != 1 || !value[0].is_finite() || !(-1.0..=1.0).contains(&value[0]) {
            return Err("invalid encoder value".into());
        }
        self.stats.encoder_seconds += tick.elapsed().as_secs_f64();
        self.stats.value_calls += 1;
        Ok(f64::from(value[0]))
    }

    fn evaluate_inner(&mut self, game: &Game) -> Result<Evaluation, Box<dyn std::error::Error>> {
        let tick = Instant::now();
        let extractor =
            FeatureExtractor::new(game, self.metadata.feature_spec.samples_per_diameter)?;
        let moves = extractor.moves(&self.actions)?;
        let resolution = extractor.resolution;
        self.stats.feature_seconds += tick.elapsed().as_secs_f64();
        let tick = Instant::now();
        let encoded = self
            .encoder
            .run(ort::inputs!["coarse" => TensorRef::from_array_view(
            ([1, COARSE_PLANES.len(), resolution, resolution], extractor.coarse.as_slice()))?])?;
        let (shape, spatial) = encoded["spatial"].try_extract_tensor::<f32>()?;
        if shape.as_ref()
            != [
                1,
                self.metadata.config.policy_hidden as i64,
                resolution as i64,
                resolution as i64,
            ]
        {
            return Err("bad encoder spatial shape".into());
        }
        let (_, pass) = encoded["pass_logit"].try_extract_tensor::<f32>()?;
        let (_, value) = encoded["value"].try_extract_tensor::<f32>()?;
        if pass.len() != 1
            || value.len() != 1
            || !pass[0].is_finite()
            || !value[0].is_finite()
            || !(-1.0..=1.0).contains(&value[0])
            || spatial.iter().any(|v| !v.is_finite())
        {
            return Err("invalid encoder outputs".into());
        }
        self.stats.encoder_seconds += tick.elapsed().as_secs_f64();
        let tick = Instant::now();
        let mut logits = Vec::with_capacity(COORDINATE_ACTIONS as usize + 1);
        let chunk = self
            .metadata
            .config
            .chunk_size
            .min(COORDINATE_ACTIONS as usize);
        for start in (0..COORDINATE_ACTIONS as usize).step_by(chunk) {
            let end = (start + chunk).min(COORDINATE_ACTIONS as usize);
            let ids: Vec<i64> = (start as i64..end as i64).collect();
            let output = self.scorer.run(ort::inputs![
                "spatial" => &encoded["spatial"],
                "exact" => TensorRef::from_array_view(([1, end-start, MOVE_FEATURES.len()],
                    &moves.exact[start*MOVE_FEATURES.len()..end*MOVE_FEATURES.len()]))?,
                "actions" => TensorRef::from_array_view(([end-start], ids.as_slice()))?
            ])?;
            let (shape, scores) = output["logits"].try_extract_tensor::<f32>()?;
            if shape.as_ref() != [1, (end - start) as i64] || scores.iter().any(|v| !v.is_finite())
            {
                return Err("invalid scorer outputs".into());
            }
            logits.extend_from_slice(scores);
        }
        for (logit, legal) in logits.iter_mut().zip(&moves.legal) {
            if *legal == 0 {
                *logit = f32::NEG_INFINITY;
            }
        }
        logits.push(pass[0]);
        self.stats.scorer_seconds += tick.elapsed().as_secs_f64();
        self.stats.calls += 1;
        Ok(Evaluation {
            value: f64::from(value[0]),
            logits,
            legal: moves.legal,
        })
    }
}

impl Evaluator for OnnxEvaluator {
    fn evaluate(&mut self, game: &Game) -> Result<Evaluation, String> {
        self.evaluate_inner(game)
            .map_err(|error| format!("ONNX evaluation: {error}"))
    }

    fn model_id(&self) -> &str {
        &self.metadata.checkpoint_sha256
    }

    fn value(&mut self, game: &Game) -> Result<f64, String> {
        self.value_inner(game)
            .map_err(|error| format!("ONNX value: {error}"))
    }
}
