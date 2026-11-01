//! Existing CGFEAT01 tensor packet, shared by one-shot and persistent extraction.

use crate::{
    features::{COARSE_PLANES, FeatureExtractor, MOVE_FEATURES, SCHEMA_VERSION},
    replay::{ReplayPosition, player_index},
};
use continugo_rules::Action;

pub fn encode_position(
    sample: &ReplayPosition,
    ply: usize,
    density: usize,
    actions: &[Action],
    full_domain: bool,
) -> Result<Vec<u8>, String> {
    let extractor = FeatureExtractor::new(&sample.game, density)?;
    let moves = extractor.moves(actions)?;
    let policy = sample.policy.as_deref().unwrap_or_default();
    let length = 64
        + extractor.coarse.len() * 4
        + moves.exact.len() * 4
        + actions.len() * 5
        + policy.len() * 8;
    let mut packet = Vec::with_capacity(length);
    packet.extend_from_slice(b"CGFEAT01");
    for number in [
        SCHEMA_VERSION,
        COARSE_PLANES.len() as u32,
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
    debug_assert_eq!(packet.len(), length);
    Ok(packet)
}
