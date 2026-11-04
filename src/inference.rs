//! Native CPU/CUDA ONNX inference. The same graphs and rules serve both devices.
//! CUDA registration is mandatory when requested; no silent CPU fallback.
//! Coarse cell proposals bound exact work; value-only leaves never score moves.

mod batch;
pub mod batching;
mod cache;
mod proposals;
pub use batch::InferenceOutput;
use cache::ContextCache;

use crate::features::{COARSE_PLANES, MOVE_FEATURES, SCHEMA_VERSION};
use continugo_rules::Game;
use ort::{
    execution_providers::{
        CUDAExecutionProvider, ExecutionProvider, cuda::CUDAExecutionProviderCuDNNConvAlgoSearch,
    },
    session::Session,
};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum InferenceDevice {
    Cpu,
    Cuda,
}
impl InferenceDevice {
    pub fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }
}

#[derive(Debug)]
pub struct Evaluation {
    pub value: f64,
    /// Hierarchical log probabilities in action-ID order, pass at 1,000,000.
    /// Unselected or illegal coordinates are -inf. Consumers normalize over
    /// selected nonempty cells and pass, not the full board.
    pub logits: Vec<f32>,
    /// Exact legality inside selected cells; zero also means unselected, not a
    /// claim of illegality across the whole board.
    pub legal: Vec<u8>,
}

pub trait Evaluator {
    fn evaluate(&mut self, game: &Game) -> Result<Evaluation, String>;
    /// Search supplies a per-node RNG seed so exploration varies across games
    /// without depending on inference queue order. Test evaluators may ignore it.
    fn evaluate_seeded(&mut self, game: &Game, _seed: u64) -> Result<Evaluation, String> {
        self.evaluate(game)
    }
    fn model_id(&self) -> &str;
    fn network_name(&self) -> &str {
        self.model_id()
    }
    fn value(&mut self, game: &Game) -> Result<f64, String> {
        Ok(self.evaluate(game)?.value)
    }
}

#[derive(Deserialize)]
struct FeatureSpec {
    schema: u32,
    policy: String,
    coarse: Vec<String>,
    moves: Vec<String>,
    samples_per_diameter: usize,
}
#[derive(Deserialize)]
struct Config {
    chunk_size: usize,
    policy_hidden: usize,
    top_cells: usize,
    exploration_cells: usize,
}
#[derive(Deserialize)]
struct Metadata {
    checkpoint_sha256: String,
    network_name: Option<String>,
    config: Config,
    feature_spec: FeatureSpec,
}

#[derive(Debug, Default, serde::Serialize)]
pub struct InferenceStats {
    pub calls: u64,
    pub value_calls: u64,
    pub feature_seconds: f64,
    pub coarse_seconds: f64,
    pub move_seconds: f64,
    pub encoder_seconds: f64,
    pub scorer_seconds: f64,
    pub encoder_positions: u64,
    pub encoder_batches: u64,
    pub encoder_max_batch: usize,
    pub scorer_batches: u64,
    pub scorer_positions: u64,
    pub scorer_max_batch: usize,
    /// Actual exact rows, excluding padding; no million-coordinate scan.
    pub exact_coordinates: u64,
    pub selected_cells: u64,
    /// Worker queue/collection delay and end-to-end request latency, summed.
    pub queue_seconds: f64,
    pub request_seconds: f64,
    pub requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_evictions: u64,
    pub cache_entries: usize,
    /// Encoded spatial + cell-logit payloads; excludes Game keys/allocator overhead.
    pub cache_bytes: usize,
}

pub struct OnnxEvaluator {
    encoder: Session,
    scorer: Session,
    metadata: Metadata,
    feature_pool: Option<rayon::ThreadPool>,
    cache: ContextCache,
    pub device: InferenceDevice,
    pub stats: InferenceStats,
}

impl OnnxEvaluator {
    pub fn samples_per_diameter(&self) -> usize {
        self.metadata.feature_spec.samples_per_diameter
    }

    pub fn load(
        directory: &Path,
        library: Option<&Path>,
        threads: usize,
        device: InferenceDevice,
    ) -> Result<Self, String> {
        Self::load_inner(directory, library, threads, device)
            .map_err(|error| format!("ONNX {} load: {error}", device.name()))
    }

    /// A single bounded CPU pool, reused for every position. Reconfiguration
    /// clears the per-model context cache, never mutating the model or rules.
    pub fn configure_features(
        &mut self,
        threads: usize,
        cache_positions: usize,
    ) -> Result<(), String> {
        if threads == 0 {
            return Err("feature thread count must be positive".into());
        }
        let pool = if threads == 1 {
            None
        } else {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .thread_name(|i| format!("features-{i}"))
                    .build()
                    .map_err(|error| format!("feature pool: {error}"))?,
            )
        };
        self.feature_pool = pool;
        self.cache = ContextCache::new(cache_positions, 32 * 1024 * 1024);
        self.stats.cache_entries = 0;
        self.stats.cache_bytes = 0;
        Ok(())
    }

    fn load_inner(
        directory: &Path,
        library: Option<&Path>,
        threads: usize,
        device: InferenceDevice,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if threads == 0 {
            return Err("ONNX thread count must be positive".into());
        }
        let library = library.map(Path::to_path_buf)
            .or_else(|| std::env::var_os("ORT_DYLIB_PATH").map(Into::into))
            .ok_or("set ORT_DYLIB_PATH or --ort-library to an ONNX Runtime >=1.20 shared library (see docs/current/inference.md)")?;
        if !library.is_file() {
            return Err(format!("runtime library does not exist: {}", library.display()).into());
        }
        ort::init_from(library.to_string_lossy())
            .with_telemetry(false)
            .commit()?;
        if device == InferenceDevice::Cuda && !CUDAExecutionProvider::default().is_available()? {
            return Err("CUDA requested but this ONNX Runtime has no CUDA execution provider; use setup-gpu or explicitly select cpu".into());
        }
        let load = |name: &str| -> ort::Result<Session> {
            let mut builder = Session::builder()?
                .with_intra_threads(threads)?
                .with_inter_threads(1)?
                .with_config_entry("session.intra_op.allow_spinning", "0")?
                .with_config_entry("session.inter_op.allow_spinning", "0")?;
            if device == InferenceDevice::Cuda {
                builder = builder.with_execution_providers([CUDAExecutionProvider::default()
                    .with_device_id(0)
                    .with_tf32(false)
                    .with_conv_algorithm_search(CUDAExecutionProviderCuDNNConvAlgoSearch::Heuristic)
                    .with_conv_max_workspace(false)
                    .with_memory_limit(2 * 1024 * 1024 * 1024)
                    .build()
                    .error_on_failure()])?;
            }
            if let Some(path) = std::env::var_os("CONTINUGO_ORT_PROFILE") {
                builder = builder.with_profiling(PathBuf::from(path).join(name))?;
            }
            builder.commit_from_file(directory.join(name))
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
        let metadata: Metadata = serde_json::from_str(&encoder_metadata).map_err(|e| {
            format!("incompatible selective-policy metadata; retrain/re-export: {e}")
        })?;
        let spec = &metadata.feature_spec;
        if spec.schema != SCHEMA_VERSION
            || spec.policy != "cells-v1"
            || spec.coarse != COARSE_PLANES
            || spec.moves != MOVE_FEATURES
            || spec.samples_per_diameter == 0
            || metadata.config.chunk_size == 0
            || metadata.config.policy_hidden == 0
            || metadata.config.top_cells == 0
            || metadata.config.exploration_cells == 0
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
            device,
            feature_pool: None,
            cache: ContextCache::new(64, 32 * 1024 * 1024),
            stats: InferenceStats::default(),
        })
    }
}

impl Evaluator for OnnxEvaluator {
    fn evaluate(&mut self, game: &Game) -> Result<Evaluation, String> {
        match self.evaluate_batch(&[(game, true)])?.pop().unwrap() {
            InferenceOutput::Policy(evaluation) => Ok(evaluation),
            InferenceOutput::Value(_) => unreachable!(),
        }
    }
    fn evaluate_seeded(&mut self, game: &Game, seed: u64) -> Result<Evaluation, String> {
        let context = self
            .contexts(&[game])
            .map_err(|e| format!("ONNX contexts: {e}"))?
            .pop()
            .unwrap();
        self.policies(&[(game, context, seed)])
            .map_err(|e| format!("ONNX policies: {e}"))?
            .pop()
            .ok_or_else(|| "missing policy".into())
    }
    fn model_id(&self) -> &str {
        &self.metadata.checkpoint_sha256
    }
    fn network_name(&self) -> &str {
        self.metadata
            .network_name
            .as_deref()
            .unwrap_or(self.model_id())
    }
    fn value(&mut self, game: &Game) -> Result<f64, String> {
        match self.evaluate_batch(&[(game, false)])?.pop().unwrap() {
            InferenceOutput::Value(value) => Ok(value),
            InferenceOutput::Policy(_) => unreachable!(),
        }
    }
}
