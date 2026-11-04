//! Version-4 side-to-move neural inputs. Rule facts use the shared validator.
//! Group liberty area matches the UI's radius-swept exact integer center spans,
//! numerically integrated in stone-disc units once per existing group.

use continugo_rules::{
    Action, COORD_RANGE, COORDINATE_ACTIONS, Game, PreparedPosition, Spot, Stone,
};

use crate::profile::Stage;
use rayon::prelude::*;
use std::time::Instant;

/// Disjoint per-extraction CPU stage totals, including shared-rule group clocks.
/// Parallel callers sum worker time, not elapsed batch time.
#[derive(Debug, Default, serde::Serialize)]
pub struct CoarseProfile {
    pub positions: u64,
    pub groups: u64,
    pub stones: u64,
    pub spans: u64,
    pub preparation: Stage,
    pub liberty_spans: Stage,
    pub capturability: Stage,
    pub liberty_area: Stage,
    pub painting: Stage,
}

impl CoarseProfile {
    pub fn merge(&mut self, other: &Self) {
        self.positions += other.positions;
        self.groups += other.groups;
        self.stones += other.stones;
        self.spans += other.spans;
        for (target, source) in [
            (&mut self.preparation, &other.preparation),
            (&mut self.liberty_spans, &other.liberty_spans),
            (&mut self.capturability, &other.capturability),
            (&mut self.liberty_area, &other.liberty_area),
            (&mut self.painting, &other.painting),
        ] {
            target.calls += source.calls;
            target.seconds += source.seconds;
        }
    }
}

pub const SCHEMA_VERSION: u32 = 4;
pub const COARSE_PLANES: [&str; 12] = [
    "friendly_stones",
    "enemy_stones",
    "friendly_group_size",
    "enemy_group_size",
    "friendly_liberty_area_stones",
    "enemy_liberty_area_stones",
    "friendly_capturable_in_one",
    "enemy_capturable_in_one",
    "left_border",
    "right_border",
    "top_border",
    "bottom_border",
];
pub const MOVE_FEATURES: [&str; 10] = [
    "legal",
    "connected_groups",
    "connected_stones",
    "captured_groups",
    "captured_stones",
    "resulting_group_size",
    "left_border",
    "right_border",
    "top_border",
    "bottom_border",
];

/// Coarse grid endpoints match the model's align_corners sampling at 0 and 999.
pub fn resolution(diameter: i32, samples_per_diameter: usize) -> Result<usize, String> {
    if samples_per_diameter == 0 || !(1..=COORD_RANGE).contains(&diameter) {
        return Err("positive sample density and diameter in 1..=1000 required".into());
    }
    let intervals = ((COORD_RANGE - 1) as usize)
        .checked_mul(samples_per_diameter)
        .ok_or("resolution overflow")?;
    let result = intervals
        .div_ceil(diameter as usize)
        .checked_add(1)
        .ok_or("resolution overflow")?;
    if result > 2048 {
        return Err("coarse resolution exceeds 2048 samples per side".into());
    }
    Ok(result)
}

/// Counts are log1p-scaled against the million-coordinate domain, without clipping.
pub fn count_feature(count: usize) -> f32 {
    (count as f32).ln_1p() / (COORDINATE_ACTIONS as f32).ln_1p()
}

pub struct MoveFeatures {
    /// Candidate-major [N, 10]; illegal tactical columns are zero.
    pub exact: Vec<f32>,
    pub legal: Vec<u8>,
}

pub struct FeatureExtractor<'a> {
    game: &'a Game,
    prepared: PreparedPosition<'a>,
    pub resolution: usize,
    /// Plane-major [12, H, W].
    pub coarse: Vec<f32>,
}

impl<'a> FeatureExtractor<'a> {
    pub fn new(game: &'a Game, samples_per_diameter: usize) -> Result<Self, String> {
        Self::build::<false>(game, samples_per_diameter).map(|(extractor, _)| extractor)
    }

    /// Diagnostic path with identical rules and rasters; ordinary extraction has
    /// no per-group clock reads. Timings exclude replay and move-row extraction.
    pub fn profiled(
        game: &'a Game,
        samples_per_diameter: usize,
    ) -> Result<(Self, CoarseProfile), String> {
        Self::build::<true>(game, samples_per_diameter)
    }

    fn build<const PROFILE: bool>(
        game: &'a Game,
        samples_per_diameter: usize,
    ) -> Result<(Self, CoarseProfile), String> {
        let mut profile = CoarseProfile::default();
        let tick = PROFILE.then(Instant::now);
        let mut this = Self::for_moves(game)?;
        if let Some(tick) = tick {
            profile.preparation.record(tick);
            profile.positions = 1;
            profile.stones = game.position().stones().len() as u64;
        }
        let board = game.board();
        let resolution = resolution(board.diameter(), samples_per_diameter)?;
        this.resolution = resolution;
        let tick = PROFILE.then(Instant::now);
        this.coarse = vec![0.0; COARSE_PLANES.len() * resolution * resolution];
        if let Some(tick) = tick {
            profile.painting.record(tick);
        }
        for color in [game.on_move(), game.on_move().other()] {
            let offset = usize::from(color != game.on_move());
            for members in this.prepared.groups(color) {
                if PROFILE {
                    profile.groups += 1;
                }
                let size = count_feature(members.len());
                let analysis = if PROFILE {
                    let (analysis, stages) = this.prepared.analyze_group_profiled(members);
                    profile.liberty_spans.calls += 1;
                    profile.liberty_spans.seconds += stages.liberty_spans_seconds;
                    profile.capturability.calls += 1;
                    profile.capturability.seconds += stages.capturability_seconds;
                    profile.spans += analysis.liberty_spans.len() as u64;
                    analysis
                } else {
                    this.prepared.analyze_group(members)
                };
                let tick = PROFILE.then(Instant::now);
                let liberty_area = crate::liberty_area::in_stone_discs(
                    &analysis.liberty_spans,
                    board.stone_radius(),
                ) as f32;
                if let Some(tick) = tick {
                    profile.liberty_area.record(tick);
                }
                let capturable = u8::from(analysis.capture_in_one.is_some()) as f32;
                let tick = PROFILE.then(Instant::now);
                for &i in members {
                    let spot = game.position().stones()[i].spot;
                    Self::paint(
                        &mut this.coarse,
                        resolution,
                        board.stone_radius(),
                        spot,
                        &[
                            (offset, 1.0),
                            (2 + offset, size),
                            (4 + offset, liberty_area),
                            (6 + offset, capturable),
                        ],
                    );
                }
                if let Some(tick) = tick {
                    profile.painting.record(tick);
                }
            }
        }
        let tick = PROFILE.then(Instant::now);
        let area = resolution * resolution;
        for y in 0..resolution {
            for x in 0..resolution {
                let spot = Self::grid_spot(resolution, x, y);
                for (direction, value) in this.borders(spot).into_iter().enumerate() {
                    this.coarse[(8 + direction) * area + y * resolution + x] = value;
                }
            }
        }
        if let Some(tick) = tick {
            profile.painting.record(tick);
        }
        Ok((this, profile))
    }

    /// Move facts only: inference already has a cached coarse-network context.
    /// Keep this crate-private so an empty raster cannot escape as a full input.
    pub(crate) fn for_moves(game: &'a Game) -> Result<Self, String> {
        if game.is_finished() {
            return Err("cannot extract a policy input from a finished game".into());
        }
        Ok(Self {
            game,
            prepared: game.position().prepare(),
            resolution: 0,
            coarse: Vec::new(),
        })
    }

    fn grid_spot(resolution: usize, x: usize, y: usize) -> Spot {
        let coordinate =
            |i| ((i * (COORD_RANGE - 1) as usize + (resolution - 1) / 2) / (resolution - 1)) as i32;
        Spot {
            x: coordinate(x),
            y: coordinate(y),
        }
    }

    fn paint(
        coarse: &mut [f32],
        resolution: usize,
        radius: i32,
        centre: Spot,
        planes: &[(usize, f32)],
    ) {
        let scale = (resolution - 1) as f64 / f64::from(COORD_RANGE - 1);
        // One extra sample covers integer rounding at the raster boundary.
        let bounds = |c: i32| {
            let start = ((f64::from(c - radius) * scale).floor() as i32 - 1).max(0) as usize;
            let end = ((f64::from(c + radius) * scale).ceil() as usize + 1).min(resolution - 1);
            start..=end
        };
        let area = resolution * resolution;
        for y in bounds(centre.y) {
            for x in bounds(centre.x) {
                let spot = Self::grid_spot(resolution, x, y);
                let distance = (spot.x - centre.x).pow(2) + (spot.y - centre.y).pow(2);
                // Physical stone disk, not the larger move-exclusion disk.
                if distance < radius * radius {
                    for &(plane, value) in planes {
                        coarse[plane * area + y * resolution + x] = value;
                    }
                }
            }
        }
    }

    fn borders(&self, spot: Spot) -> [f32; 4] {
        let board = self.game.board();
        let low = board.stone_radius();
        let high = COORD_RANGE - low;
        [spot.x - low, high - spot.x, spot.y - low, high - spot.y]
            .map(|distance| (distance as f32 / board.diameter() as f32).clamp(0.0, 4.0) / 4.0)
    }

    /// Any coordinate subset, in caller order. Full-domain callers use IDs
    /// 0..1,000,000; pass has a separate pooled head and is never a feature row.
    pub fn moves(&self, actions: &[Action]) -> Result<MoveFeatures, String> {
        let mut result = MoveFeatures {
            exact: vec![0.0; actions.len() * MOVE_FEATURES.len()],
            legal: vec![0; actions.len()],
        };
        self.fill_moves(actions, &mut result.exact, &mut result.legal)?;
        Ok(result)
    }

    /// Bounded caller-owned CPU pool, sharing one immutable prepared rules view.
    /// Disjoint output rows preserve action order and exactly match serial facts.
    pub fn moves_parallel(
        &self,
        actions: &[Action],
        pool: &rayon::ThreadPool,
    ) -> Result<MoveFeatures, String> {
        const CHUNK: usize = 4096;
        let mut result = MoveFeatures {
            exact: vec![0.0; actions.len() * MOVE_FEATURES.len()],
            legal: vec![0; actions.len()],
        };
        pool.install(|| {
            result
                .exact
                .par_chunks_mut(CHUNK * MOVE_FEATURES.len())
                .zip(result.legal.par_chunks_mut(CHUNK))
                .zip(actions.par_chunks(CHUNK))
                .try_for_each(|((exact, legal), actions)| self.fill_moves(actions, exact, legal))
        })?;
        Ok(result)
    }

    fn fill_moves(
        &self,
        actions: &[Action],
        exact: &mut [f32],
        legal: &mut [u8],
    ) -> Result<(), String> {
        for ((&action, row), legal) in actions
            .iter()
            .zip(exact.chunks_mut(MOVE_FEATURES.len()))
            .zip(legal.iter_mut())
        {
            let spot = action
                .spot()
                .ok_or("pass must not appear in coordinate features")?;
            row[6..10].copy_from_slice(&self.borders(spot));
            let Some(analysis) = self.prepared.analyze_move(
                Stone {
                    color: self.game.on_move(),
                    spot,
                },
                0, // No per-candidate area integration or unused liberty-count search.
            ) else {
                continue;
            };
            *legal = 1;
            row[..6].copy_from_slice(&[
                1.0,
                count_feature(analysis.connected_groups),
                count_feature(analysis.connected_stones),
                count_feature(analysis.captured_groups),
                count_feature(analysis.captured_stones),
                count_feature(analysis.group_size),
            ]);
        }
        Ok(())
    }
}
