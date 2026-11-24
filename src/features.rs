//! Version-7 inputs: recent-stone coverage plus coarse width / 100,
//! derived from raster shape inside the encoder as an unclipped global scalar.
//! Schemas 4–6 retain their separately versioned inputs for inference.

use continugo_rules::{
    Action, COORD_RANGE, COORDINATE_ACTIONS, Game, PreparedPosition, Spot, Stone,
};

mod raster;
pub mod schema5;
pub mod schema6;
pub mod schema7;
mod spatial;
mod splitting;

use crate::{
    geometry_proposals::{Geometry, GroupGeometry},
    profile::Stage,
};
use rayon::prelude::*;
use std::time::Instant;

/// Per-extraction CPU stage totals, including shared-rule group clocks.
/// `raster` subdivides `painting`; do not add it to the top-level stages.
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
    pub raster: RasterProfile,
}

/// Disjoint painting subtotals; profiling/bookkeeping overhead is excluded.
#[derive(Debug, Default, serde::Serialize)]
pub struct RasterProfile {
    pub allocation: Stage,
    /// Lazy allocation/zeroing of the independent group size/area canvases.
    pub group_canvas_allocation: Stage,
    /// Shared coverage/painting for group size and liberty area. Count once,
    /// not per plane. Canvas allocation is separate.
    pub group_shared_shapes: Stage,
    pub occupancy: Stage,
    pub history: Stage,
    pub liberty_union: Stage,
    pub group_reduce: Stage,
    pub group_liberty_reduce: Stage,
    pub liberty_centers: Stage,
    pub liberty_footprints: Stage,
    pub borders: Stage,
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
            (&mut self.raster.allocation, &other.raster.allocation),
            (
                &mut self.raster.group_canvas_allocation,
                &other.raster.group_canvas_allocation,
            ),
            (
                &mut self.raster.group_shared_shapes,
                &other.raster.group_shared_shapes,
            ),
            (&mut self.raster.occupancy, &other.raster.occupancy),
            (&mut self.raster.history, &other.raster.history),
            (&mut self.raster.liberty_union, &other.raster.liberty_union),
            (&mut self.raster.group_reduce, &other.raster.group_reduce),
            (
                &mut self.raster.group_liberty_reduce,
                &other.raster.group_liberty_reduce,
            ),
            (
                &mut self.raster.liberty_centers,
                &other.raster.liberty_centers,
            ),
            (
                &mut self.raster.liberty_footprints,
                &other.raster.liberty_footprints,
            ),
            (&mut self.raster.borders, &other.raster.borders),
        ] {
            target.calls += source.calls;
            target.seconds += source.seconds;
        }
    }
}

// Schema 7 remains open until the next fresh run starts.
pub use schema7::{
    COARSE_PLANES, MOVE_FEATURES, VERSION as SCHEMA_VERSION, count_feature, scalar_feature,
};
pub const POINT_SAMPLED_SCHEMA: u32 = 4;

pub fn validate_schema(schema: u32) -> Result<(), String> {
    if matches!(
        schema,
        POINT_SAMPLED_SCHEMA | schema5::VERSION | schema6::VERSION | schema7::VERSION
    ) {
        Ok(())
    } else {
        Err(format!("unsupported feature schema {schema}"))
    }
}
pub fn coarse_planes(schema: u32) -> Result<&'static [&'static str], String> {
    validate_schema(schema)?;
    Ok(match schema {
        schema5::VERSION => &schema5::COARSE_PLANES,
        schema6::VERSION => &schema6::COARSE_PLANES,
        schema7::VERSION => &schema7::COARSE_PLANES,
        _ => &SCHEMA4_COARSE_PLANES,
    })
}

pub const SCHEMA4_COARSE_PLANES: [&str; 12] = [
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
pub const SCHEMA4_MOVE_FEATURES: [&str; 10] = [
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

pub fn move_features(schema: u32) -> Result<&'static [&'static str], String> {
    validate_schema(schema)?;
    Ok(match schema {
        schema5::VERSION => &schema5::MOVE_FEATURES,
        schema6::VERSION => &schema6::MOVE_FEATURES,
        schema7::VERSION => &schema7::MOVE_FEATURES,
        _ => &SCHEMA4_MOVE_FEATURES,
    })
}

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

// Schemas 4 and 5 use the same uncapped normalization.
fn schema45_count_feature(count: usize) -> f32 {
    (count as f32).ln_1p() / (COORDINATE_ACTIONS as f32).ln_1p()
}

pub struct MoveFeatures {
    /// Candidate-major [N, fields]: schema 4 has 10, schema 5 has 11, schemas 6/7 have 9.
    /// Illegal tactical/splitting columns are zero.
    pub exact: Vec<f32>,
    pub legal: Vec<u8>,
}

pub struct FeatureExtractor<'a> {
    game: &'a Game,
    prepared: PreparedPosition<'a>,
    move_fields: usize,
    splitting: splitting::Cache,
    pub resolution: usize,
    /// Plane-major [planes, H, W]: schemas 4/5 have 12, schema 6 has 14, schema 7 has 17.
    pub coarse: Vec<f32>,
    /// Opt-in exact per-group source geometry, before colour union/rasterization.
    pub geometry: Option<std::sync::Arc<Geometry>>,
}

impl<'a> FeatureExtractor<'a> {
    pub fn new(game: &'a Game, samples_per_diameter: usize) -> Result<Self, String> {
        Self::with_schema(game, samples_per_diameter, SCHEMA_VERSION)
    }

    /// Preserve point-sampled inputs for explicitly versioned existing models.
    pub fn with_schema(game: &'a Game, density: usize, schema: u32) -> Result<Self, String> {
        Self::build::<false>(game, density, schema, false).map(|(extractor, _)| extractor)
    }

    /// Diagnostic path with identical rules and rasters; ordinary extraction has
    /// no per-group clock reads. Timings exclude replay and move-row extraction.
    pub fn profiled(
        game: &'a Game,
        samples_per_diameter: usize,
    ) -> Result<(Self, CoarseProfile), String> {
        Self::profiled_with_schema(game, samples_per_diameter, SCHEMA_VERSION)
    }

    pub fn profiled_with_schema(
        game: &'a Game,
        density: usize,
        schema: u32,
    ) -> Result<(Self, CoarseProfile), String> {
        Self::build::<true>(game, density, schema, false)
    }

    /// Keep exact geometry already computed for current coarse inputs.
    pub fn with_geometry(
        game: &'a Game,
        density: usize,
        schema: u32,
        profile: bool,
    ) -> Result<(Self, CoarseProfile), String> {
        if !matches!(
            schema,
            schema5::VERSION | schema6::VERSION | schema7::VERSION
        ) {
            return Err("geometry proposals require feature schema 5 through 7".into());
        }
        if profile {
            Self::build::<true>(game, density, schema, true)
        } else {
            Self::build::<false>(game, density, schema, true)
        }
    }

    fn build<const PROFILE: bool>(
        game: &'a Game,
        samples_per_diameter: usize,
        schema: u32,
        retain_geometry: bool,
    ) -> Result<(Self, CoarseProfile), String> {
        validate_schema(schema)?;
        if schema == schema5::VERSION {
            return schema5::build::<PROFILE>(game, samples_per_diameter, retain_geometry);
        }
        if schema == schema7::VERSION {
            return schema7::build::<PROFILE>(game, samples_per_diameter, retain_geometry);
        }
        let mut profile = CoarseProfile::default();
        let tick = PROFILE.then(Instant::now);
        let mut this = Self::for_moves(game, schema)?;
        if let Some(tick) = tick {
            profile.preparation.record(tick);
            profile.positions = 1;
            profile.stones = game.position().stones().len() as u64;
        }
        let board = game.board();
        let resolution = resolution(board.diameter(), samples_per_diameter)?;
        this.resolution = resolution;
        let edges = crate::targets::raster_bounds(resolution);
        let area = resolution * resolution;
        let tick = PROFILE.then(Instant::now);
        let planes = coarse_planes(schema)?.len();
        this.coarse = vec![0.0; planes * resolution * resolution];
        if let Some(tick) = tick {
            profile.painting.record(tick);
            profile.raster.allocation.record(tick);
        }
        let grid = (schema == schema6::VERSION).then(|| spatial::Grid::new(&edges));
        let mut sizes: [Option<Vec<f32>>; 2] = [None, None];
        let mut group_areas: [Option<Vec<f32>>; 2] = [None, None];
        let mut liberties = [Vec::new(), Vec::new()];
        let mut geometry = retain_geometry.then(Geometry::default);
        for color in [game.on_move(), game.on_move().other()] {
            let offset = usize::from(color != game.on_move());
            for members in this.prepared.groups(color) {
                if PROFILE {
                    profile.groups += 1;
                }
                let size = if schema == schema6::VERSION {
                    count_feature(members.len())
                } else {
                    schema45_count_feature(members.len())
                };
                if let Some(grid) = &grid {
                    let tick = PROFILE.then(Instant::now);
                    let spans = this.prepared.liberty_spans(members);
                    if let Some(tick) = tick {
                        profile.liberty_spans.record(tick);
                        profile.spans += spans.len() as u64;
                    }
                    let tick = PROFILE.then(Instant::now);
                    let liberty_area = crate::liberty_area::coarse_input_in_stone_discs(
                        &spans,
                        board.stone_radius(),
                    );
                    if let Some(tick) = tick {
                        profile.liberty_area.record(tick);
                    }
                    let tick = PROFILE.then(Instant::now);
                    let sub = PROFILE.then(Instant::now);
                    let size_canvas = sizes[offset].get_or_insert_with(|| grid.canvas());
                    let area_canvas = group_areas[offset].get_or_insert_with(|| grid.canvas());
                    if let Some(sub) = sub {
                        profile.raster.group_canvas_allocation.record(sub);
                    }
                    let sub = PROFILE.then(Instant::now);
                    grid.group_values(
                        &mut [size_canvas.as_mut_slice(), area_canvas.as_mut_slice()],
                        board,
                        game.position().stones(),
                        members,
                        [size, scalar_feature(liberty_area)],
                    );
                    if let Some(sub) = sub {
                        profile.raster.group_shared_shapes.record(sub);
                    }
                    if let Some(geometry) = &mut geometry {
                        geometry.groups.push(GroupGeometry {
                            friendly: offset == 0,
                            members: members.to_vec(),
                            spans: spans.clone(),
                        });
                    }
                    liberties[offset].extend(spans);
                    if let Some(tick) = tick {
                        profile.painting.record(tick);
                    }
                    continue;
                }
                // Legacy schema 4: keep every original feature bit unchanged.
                let analysis = if PROFILE {
                    let (analysis, stages) = this.prepared.analyze_group_profiled(members);
                    profile.liberty_spans.calls += 1;
                    profile.liberty_spans.seconds += stages.liberty_spans_seconds;
                    profile.capturability.calls += 1;
                    profile.capturability.seconds += stages.capturability_seconds;
                    analysis
                } else {
                    this.prepared.analyze_group(members)
                };
                let (spans, capturable) = (
                    analysis.liberty_spans,
                    u8::from(analysis.capture_in_one.is_some()) as f32,
                );
                if PROFILE {
                    profile.spans += spans.len() as u64;
                }
                let tick = PROFILE.then(Instant::now);
                let liberty_area =
                    crate::liberty_area::in_stone_discs(&spans, board.stone_radius()) as f32;
                if let Some(tick) = tick {
                    profile.liberty_area.record(tick);
                }
                let tick = PROFILE.then(Instant::now);
                for &i in members {
                    let spot = game.position().stones()[i].spot;
                    let fields = [
                        (offset, 1.0),
                        (2 + offset, size),
                        (4 + offset, liberty_area),
                        (6 + offset, capturable),
                    ];
                    let fields = &fields[..];
                    Self::paint(
                        &mut this.coarse,
                        resolution,
                        board.stone_radius(),
                        spot,
                        fields,
                    );
                }
                if let Some(tick) = tick {
                    profile.painting.record(tick);
                }
            }
        }
        if let Some(grid) = &grid {
            let tick = PROFILE.then(Instant::now);
            // Stable geometric order makes float accumulation independent of
            // move history/group enumeration for the same physical position.
            let mut stones: Vec<_> = game.position().stones().iter().collect();
            stones.sort_unstable_by_key(|stone| (stone.spot.x, stone.spot.y));
            for stone in stones {
                let offset = usize::from(stone.color != game.on_move());
                raster::paint(
                    &mut this.coarse[offset * area..(offset + 1) * area],
                    board.stone_radius(),
                    stone.spot,
                    &edges,
                );
            }
            if let Some(tick) = tick {
                profile.raster.occupancy.record(tick);
            }
            for (offset, spans) in liberties.iter_mut().enumerate() {
                let Some(canvas) = &sizes[offset] else {
                    continue;
                };
                let sub = PROFILE.then(Instant::now);
                let spans = spatial::union_spans(std::mem::take(spans));
                if let Some(sub) = sub {
                    profile.raster.liberty_union.record(sub);
                }
                let sub = PROFILE.then(Instant::now);
                let group = grid.reduce(canvas);
                if let Some(sub) = sub {
                    profile.raster.group_reduce.record(sub);
                }
                let sub = PROFILE.then(Instant::now);
                let centers = spatial::centers(&spans, &edges);
                if let Some(sub) = sub {
                    profile.raster.liberty_centers.record(sub);
                }
                let sub = PROFILE.then(Instant::now);
                let footprints = grid.footprints(&spans, board.stone_radius());
                if let Some(sub) = sub {
                    profile.raster.liberty_footprints.record(sub);
                }
                let sub = PROFILE.then(Instant::now);
                let group_liberties = grid.reduce(group_areas[offset].as_ref().unwrap());
                if let Some(sub) = sub {
                    profile.raster.group_liberty_reduce.record(sub);
                }
                for (plane, values) in [
                    (2 + offset, group),
                    (4 + offset, group_liberties),
                    (6 + offset, centers),
                    (8 + offset, footprints),
                ] {
                    this.coarse[plane * area..(plane + 1) * area].copy_from_slice(&values);
                }
            }
            if let Some(tick) = tick {
                profile.painting.record(tick);
            }
        }
        let tick = PROFILE.then(Instant::now);
        for y in 0..resolution {
            for x in 0..resolution {
                let spot = Self::grid_spot(resolution, x, y);
                for (direction, value) in this.borders(spot).into_iter().enumerate() {
                    this.coarse[(planes - 4 + direction) * area + y * resolution + x] = value;
                }
            }
        }
        if let Some(tick) = tick {
            profile.painting.record(tick);
            profile.raster.borders.record(tick);
        }
        this.geometry = geometry.map(std::sync::Arc::new);
        Ok((this, profile))
    }

    /// Move facts only: inference already has a cached coarse-network context.
    /// Keep this crate-private so an empty raster cannot escape as a full input.
    pub(crate) fn for_moves(game: &'a Game, schema: u32) -> Result<Self, String> {
        if game.is_finished() {
            return Err("cannot extract a policy input from a finished game".into());
        }
        Ok(Self {
            game,
            prepared: game.position().prepare(),
            move_fields: move_features(schema)?.len(),
            splitting: splitting::Cache::default(),
            resolution: 0,
            coarse: Vec::new(),
            geometry: None,
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
            exact: vec![0.0; actions.len() * self.move_fields],
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
            exact: vec![0.0; actions.len() * self.move_fields],
            legal: vec![0; actions.len()],
        };
        pool.install(|| {
            result
                .exact
                .par_chunks_mut(CHUNK * self.move_fields)
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
        let fields = self.move_fields;
        for ((&action, row), legal) in actions
            .iter()
            .zip(exact.chunks_mut(fields))
            .zip(legal.iter_mut())
        {
            let spot = action
                .spot()
                .ok_or("pass must not appear in coordinate features")?;
            row[fields - 4..].copy_from_slice(&self.borders(spot));
            let Some(analysis) = self.prepared.analyze_move(
                Stone {
                    color: self.game.on_move(),
                    spot,
                },
                0, // Splitting uses actual child survivors, not group liberty counts.
            ) else {
                continue;
            };
            // Only single-stone captures can violate ko. Keep the shared game
            // validator authoritative without replaying ordinary placements twice.
            if analysis.captured_stones == 1 && self.game.play(spot).is_err() {
                continue;
            }
            *legal = 1;
            if fields == MOVE_FEATURES.len() {
                row[..4].copy_from_slice(&[
                    count_feature(analysis.connected_groups),
                    count_feature(analysis.connected_stones),
                    count_feature(analysis.captured_groups),
                    count_feature(analysis.captured_stones),
                ]);
            } else {
                let count = schema45_count_feature;
                row[..6].copy_from_slice(&[
                    1.0,
                    count(analysis.connected_groups),
                    count(analysis.connected_stones),
                    count(analysis.captured_groups),
                    count(analysis.captured_stones),
                    count(analysis.group_size),
                ]);
            }
            if fields == schema5::MOVE_FEATURES.len() || fields == MOVE_FEATURES.len() {
                let column = if fields == schema5::MOVE_FEATURES.len() {
                    6
                } else {
                    4
                };
                row[column] = self.splitting.value(self.game, spot, &analysis.position);
            }
        }
        Ok(())
    }
}
