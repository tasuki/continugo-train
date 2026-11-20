//! Schema 7: schema-6 geometry with three recent-move markers after occupancy.

use super::{CoarseProfile, FeatureExtractor, raster, schema6};
use continugo_rules::{Game, Play};
use std::time::Instant;

pub use schema6::{MOVE_FEATURES, count_feature, scalar_feature};
pub const VERSION: u32 = 7;
pub const COARSE_PLANES: [&str; 17] = [
    "friendly_stones",
    "enemy_stones",
    "enemy_move_1_ago_coverage",
    "friendly_move_2_ago_coverage",
    "enemy_move_3_ago_coverage",
    "friendly_group_size",
    "enemy_group_size",
    "friendly_group_liberty_area",
    "enemy_group_liberty_area",
    "friendly_liberty_centers",
    "enemy_liberty_centers",
    "friendly_liberty_footprints",
    "enemy_liberty_footprints",
    "left_border",
    "right_border",
    "top_border",
    "bottom_border",
];

pub(super) fn build<const PROFILE: bool>(
    game: &Game,
    density: usize,
    retain_geometry: bool,
) -> Result<(FeatureExtractor<'_>, CoarseProfile), String> {
    // Reuse the frozen schema-6 geometry/exact contract, including retained
    // proposal geometry. History only changes the coarse encoder inputs.
    let (mut extractor, mut profile) =
        FeatureExtractor::build::<PROFILE>(game, density, schema6::VERSION, retain_geometry)?;
    let tick = PROFILE.then(Instant::now);
    let area = extractor.resolution.pow(2);
    extractor.coarse.resize(COARSE_PLANES.len() * area, 0.0);
    extractor.coarse.copy_within(2 * area..14 * area, 5 * area);
    extractor.coarse[2 * area..5 * area].fill(0.0);
    let edges = crate::targets::raster_bounds(extractor.resolution);
    for (lag, play) in game.record().iter().rev().take(3).enumerate() {
        if let Play::Place(stone) = play {
            // Same analytic physical-disc coverage as occupancy: retain the
            // recorded subcell location, not just its nearest coarse cell.
            raster::paint(
                &mut extractor.coarse[(2 + lag) * area..(3 + lag) * area],
                game.board().stone_radius(),
                stone.spot,
                &edges,
            );
        }
        // A pass consumes its slot; absent history stays zero. Never search
        // surviving stones, since an earlier placement may have been captured.
    }
    if let Some(tick) = tick {
        profile.painting.record(tick);
        profile.raster.history.record(tick);
    }
    Ok((extractor, profile))
}
