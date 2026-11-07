//! CGFEAT02 tensor packet, including validated final score/ownership targets.

use crate::{
    features::{FeatureExtractor, MOVE_FEATURES, SCHEMA_VERSION, coarse_planes},
    replay::{ReplayPosition, player_index},
};
use continugo_rules::{Action, COORDINATE_ACTIONS};
use std::time::Instant;

/// Disjoint CPU stages, excluding replay, target-cell selection and pipe writes.
#[derive(Default, serde::Serialize)]
pub struct FeatureTimings {
    pub coarse_seconds: f64,
    pub exact_seconds: f64,
    pub serialization_seconds: f64,
}

/// Only cells with positive placement visits need exact features for hierarchical
/// training. The coarse loss still normalizes over every cell and pass.
pub fn target_cell_actions(sample: &ReplayPosition, density: usize) -> Result<Vec<Action>, String> {
    let resolution = crate::features::resolution(sample.game.board().diameter(), density)?;
    let cells: Vec<_> = sample
        .policy
        .as_deref()
        .unwrap_or_default()
        .iter()
        .filter(|target| target.visits > 0 && target.action < COORDINATE_ACTIONS)
        .map(|target| crate::cells::cell(target.action, resolution))
        .collect();
    Ok(crate::cells::actions(&cells, resolution))
}

pub fn encode_position(
    sample: &ReplayPosition,
    ply: usize,
    density: usize,
    actions: &[Action],
    full_domain: bool,
) -> Result<Vec<u8>, String> {
    encode::<false>(sample, ply, density, actions, full_domain, SCHEMA_VERSION)
        .map(|(packet, _)| packet)
}

pub fn encode_position_profiled(
    sample: &ReplayPosition,
    ply: usize,
    density: usize,
    actions: &[Action],
    full_domain: bool,
) -> Result<(Vec<u8>, FeatureTimings), String> {
    encode::<true>(sample, ply, density, actions, full_domain, SCHEMA_VERSION)
}

pub fn encode_position_schema(
    sample: &ReplayPosition,
    ply: usize,
    density: usize,
    actions: &[Action],
    full_domain: bool,
    schema: u32,
    profile: bool,
) -> Result<(Vec<u8>, FeatureTimings), String> {
    if profile {
        encode::<true>(sample, ply, density, actions, full_domain, schema)
    } else {
        encode::<false>(sample, ply, density, actions, full_domain, schema)
    }
}

fn encode<const PROFILE: bool>(
    sample: &ReplayPosition,
    ply: usize,
    density: usize,
    actions: &[Action],
    full_domain: bool,
    schema: u32,
) -> Result<(Vec<u8>, FeatureTimings), String> {
    let mut timings = FeatureTimings::default();
    let tick = PROFILE.then(Instant::now);
    let extractor = FeatureExtractor::with_schema(&sample.game, density, schema)?;
    if let Some(tick) = tick {
        timings.coarse_seconds = tick.elapsed().as_secs_f64();
    }
    let tick = PROFILE.then(Instant::now);
    let moves = extractor.moves(actions)?;
    if let Some(tick) = tick {
        timings.exact_seconds = tick.elapsed().as_secs_f64();
    }
    let tick = PROFILE.then(Instant::now);
    let policy = sample.policy.as_deref().unwrap_or_default();
    let ownership = sample
        .final_targets
        .as_ref()
        .map(|t| t.ownership(extractor.resolution));
    let sign = if player_index(sample.game.on_move()) == 0 {
        1.0
    } else {
        -1.0
    };
    let score = sample
        .final_targets
        .as_ref()
        .map_or(0.0, |t| t.score * sign);
    let length = 68
        + extractor.resolution * extractor.resolution * 4
        + extractor.coarse.len() * 4
        + moves.exact.len() * 4
        + actions.len() * 5
        + policy.len() * 8;
    let mut packet = Vec::with_capacity(length);
    packet.extend_from_slice(b"CGFEAT02");
    for number in [
        schema,
        coarse_planes(schema)?.len() as u32,
        MOVE_FEATURES.len() as u32,
        sample.game.board().size() as u32,
        sample.game.board().diameter() as u32,
        extractor.resolution as u32,
        u32::try_from(ply).map_err(|_| "ply exceeds packet range")?,
        player_index(sample.game.on_move()),
        actions.len() as u32,
        policy.len() as u32,
        u32::from(sample.outcome.is_some()),
        sample.outcome.unwrap_or(0.0).to_bits(),
        density as u32,
        u32::from(full_domain),
    ] {
        packet.extend_from_slice(&number.to_le_bytes());
    }
    packet.extend_from_slice(&score.to_le_bytes());
    for &number in extractor.coarse.iter().chain(&moves.exact) {
        packet.extend_from_slice(&number.to_le_bytes());
    }
    packet.extend_from_slice(&moves.legal);
    for action in actions {
        packet.extend_from_slice(&action.index().to_le_bytes());
    }
    for target in policy {
        packet.extend_from_slice(&target.action.to_le_bytes());
    }
    for target in policy {
        packet.extend_from_slice(&target.visits.to_le_bytes());
    }
    for i in 0..extractor.resolution * extractor.resolution {
        let value = ownership.as_ref().map_or(0.0, |raster| raster[i] * sign);
        packet.extend_from_slice(&value.to_le_bytes());
    }
    debug_assert_eq!(packet.len(), length);
    if let Some(tick) = tick {
        timings.serialization_seconds = tick.elapsed().as_secs_f64();
    }
    Ok((packet, timings))
}
