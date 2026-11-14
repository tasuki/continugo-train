//! Native CPU/CUDA ONNX inference. The same graphs and rules serve both devices.
//! CUDA registration is mandatory when requested; no silent CPU fallback.
//! Coarse cell proposals bound exact work; value-only leaves never score moves.

mod batch;
pub mod batching;
mod cache;
mod normalization;
mod proposals;
pub use batch::InferenceOutput;
use cache::ContextCache;

use crate::features::{coarse_planes, move_features};
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
    pub score: f64,
    /// Inference log probabilities in action-ID order, pass at 1,000,000.
    /// Unselected/illegal coordinates are -inf. Selected nonempty cells share
    /// the full placement mass, preserving the global coarse P(pass).
    /// With no selected legal placement, pass has probability one.
    pub logits: Vec<f32>,
    /// Exact legality at analyzed coordinates; zero also means unselected or
    /// unchecked (including pruned coordinates inside geometry-mode cells).
    pub legal: Vec<u8>,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct LeafEstimate {
    pub value: f64,
    pub score: f64,
}

impl LeafEstimate {
    pub fn utility(self, weight: f64) -> Result<f64, String> {
        if !weight.is_finite() || !(0.0..=1.0).contains(&weight) {
            return Err("score weight must be finite and in [0,1]".into());
        }
        if [self.value, self.score]
            .iter()
            .any(|v| !v.is_finite() || !(-1.0..=1.0).contains(v))
        {
            return Err(format!(
                "invalid neural outcome/score estimate: value={}, score={}",
                self.value, self.score
            ));
        }
        Ok((1.0 - weight) * self.value + weight * self.score)
    }
}

#[cfg(test)]
mod utility_tests {
    use super::*;

    #[test]
    fn score_weight_is_a_linear_fraction_with_equal_weight_at_half() {
        let estimate = LeafEstimate {
            value: 0.8,
            score: -0.2,
        };
        assert!((estimate.utility(0.5).unwrap() - 0.3).abs() < 1e-12);
        assert_eq!(estimate.utility(1.0).unwrap(), estimate.score);
        assert_eq!(estimate.utility(0.0).unwrap(), estimate.value);
        for invalid in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
            assert!(estimate.utility(invalid).is_err());
        }
    }
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
    fn estimate(&mut self, game: &Game) -> Result<LeafEstimate, String> {
        let evaluation = self.evaluate(game)?;
        Ok(LeafEstimate {
            value: evaluation.value,
            score: evaluation.score,
        })
    }
    fn value(&mut self, game: &Game) -> Result<f64, String> {
        Ok(self.estimate(game)?.value)
    }
}

#[derive(Deserialize)]
struct FeatureSpec {
    schema: u32,
    policy: String,
    targets: String,
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
    pub coarse_stages: crate::features::CoarseProfile,
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
    pub proposal_seconds: f64,
    pub proposal_preliminary: u64,
    pub proposal_overflow: u64,
    /// Disjoint proposal-source/shared stages; populated only with profile_coarse.
    pub proposal_stages: crate::geometry_proposals::ProposalProfile,
    /// Worker queue/collection delay and end-to-end request latency, summed.
    pub queue_seconds: f64,
    pub request_seconds: f64,
    pub requests: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_evictions: u64,
    pub cache_entries: usize,
    /// Encoded context + optional proposal geometry; excludes Game keys/allocator overhead.
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
    /// Opt-in coarse-group and geometry-proposal stage clocks, independent of cache.
    pub profile_coarse: bool,
    geometry_budget: Option<usize>,
}

impl OnnxEvaluator {
    pub fn feature_schema(&self) -> u32 {
        self.metadata.feature_spec.schema
    }

    pub fn samples_per_diameter(&self) -> usize {
        self.metadata.feature_spec.samples_per_diameter
    }

    pub fn geometry_budget(&self) -> Option<usize> {
        self.geometry_budget
    }

    pub fn proposal_policy(&self) -> &'static str {
        if self.geometry_budget.is_some() {
            crate::geometry_proposals::VERSION
        } else {
            "complete-cells-v1"
        }
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

    /// Select geometry proposals with a budget, or complete cells with None.
    /// Changing modes clears contexts so geometry is never missing on a cache hit.
    pub fn configure_geometry(&mut self, budget: Option<usize>) -> Result<(), String> {
        if budget.is_some_and(|b| b == 0 || b > 128) {
            return Err("geometry budget must be in 1..=128".into());
        }
        if budget.is_some() && self.feature_schema() != crate::features::SCHEMA_VERSION {
            return Err("geometry proposals require feature schema 5".into());
        }
        if self.geometry_budget != budget {
            self.cache.clear();
            self.stats.cache_entries = 0;
            self.stats.cache_bytes = 0;
        }
        self.geometry_budget = budget;
        Ok(())
    }

    /// The same candidate generator/cell selection as policy inference, for dots
    /// and coverage diagnostics. No fine features or search are run here.
    pub fn geometry_shortlist(
        &mut self,
        game: &Game,
        seed: u64,
    ) -> Result<crate::geometry_proposals::Shortlist, String> {
        let budget = self
            .geometry_budget
            .ok_or("enable geometry proposals first")?;
        let context = self
            .contexts(&[game])
            .map_err(|e| e.to_string())?
            .pop()
            .unwrap();
        let selected = proposals::select(
            game,
            &context.cells,
            self.metadata.config.top_cells,
            self.metadata.config.exploration_cells,
            seed,
        );
        let started = std::time::Instant::now();
        let generate = if self.profile_coarse {
            crate::geometry_proposals::generate_profiled
        } else {
            crate::geometry_proposals::generate
        };
        let shortlist = generate(
            game,
            context.geometry.as_ref().ok_or("missing geometry")?,
            context.resolution,
            &selected,
            budget,
            seed,
        )?;
        self.stats.proposal_seconds += started.elapsed().as_secs_f64();
        if let Some(profile) = &shortlist.profile {
            self.stats.proposal_stages.merge(profile);
        }
        self.stats.proposal_preliminary += shortlist.preliminary_candidates as u64;
        self.stats.proposal_overflow += shortlist.protected_overflow as u64;
        self.stats.selected_cells += selected.len() as u64;
        Ok(shortlist)
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
        let planes = coarse_planes(spec.schema)?;
        if spec.policy != "cells-v1"
            || spec.coarse != planes
            || spec.moves != move_features(spec.schema)?
            || spec.samples_per_diameter == 0
            || spec.targets != crate::targets::TARGET_VERSION
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
        // Schema 4 retains its original complete-cell behavior. Current models
        // use direct geometry proposals unless explicitly reconfigured.
        let geometry_budget = (spec.schema == crate::features::SCHEMA_VERSION)
            .then_some(crate::geometry_proposals::DEFAULT_BUDGET);
        Ok(Self {
            encoder,
            scorer,
            metadata,
            device,
            feature_pool: None,
            cache: ContextCache::new(64, 32 * 1024 * 1024),
            stats: InferenceStats::default(),
            profile_coarse: false,
            geometry_budget,
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
    fn estimate(&mut self, game: &Game) -> Result<LeafEstimate, String> {
        match self.evaluate_batch(&[(game, false)])?.pop().unwrap() {
            InferenceOutput::Value(value) => Ok(value),
            InferenceOutput::Policy(_) => unreachable!(),
        }
    }
}
