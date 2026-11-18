//! Schema 5 input contract, recovered from Git commit 71f8daf (before schema 6).
//! Keep its uncapped counts, 0.30-radius bridges, channel order and exact splitting.

use super::{CoarseProfile, FeatureExtractor, raster, spatial};
use crate::geometry_proposals::{Geometry, GroupGeometry};
use continugo_rules::Game;
use std::time::Instant;

pub const VERSION: u32 = 5;
pub const COARSE_PLANES: [&str; 12] = [
    "friendly_stones",
    "enemy_stones",
    "friendly_group_size",
    "enemy_group_size",
    "friendly_liberty_centers",
    "enemy_liberty_centers",
    "friendly_liberty_footprints",
    "enemy_liberty_footprints",
    "left_border",
    "right_border",
    "top_border",
    "bottom_border",
];
pub const MOVE_FEATURES: [&str; 11] = [
    "legal",
    "connected_groups",
    "connected_stones",
    "captured_groups",
    "captured_stones",
    "resulting_group_size",
    "splits_friendly_region",
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
    let mut profile = CoarseProfile::default();
    let tick = PROFILE.then(Instant::now);
    let mut extractor = FeatureExtractor::for_moves(game, VERSION)?;
    if let Some(tick) = tick {
        profile.preparation.record(tick);
        profile.positions = 1;
        profile.stones = game.position().stones().len() as u64;
    }
    let board = game.board();
    let n = super::resolution(board.diameter(), density)?;
    extractor.resolution = n;
    let area = n * n;
    let edges = crate::targets::raster_bounds(n);
    let grid = spatial::Grid::with_bridge_radius(&edges, 0.30);
    extractor.coarse = vec![0.0; COARSE_PLANES.len() * area];
    let mut sizes: [Option<Vec<f32>>; 2] = [None, None];
    let mut liberties = [Vec::new(), Vec::new()];
    let mut geometry = retain_geometry.then(Geometry::default);
    for color in [game.on_move(), game.on_move().other()] {
        let offset = usize::from(color != game.on_move());
        for members in extractor.prepared.groups(color) {
            if PROFILE {
                profile.groups += 1;
            }
            let tick = PROFILE.then(Instant::now);
            let spans = extractor.prepared.liberty_spans(members);
            if let Some(tick) = tick {
                profile.liberty_spans.record(tick);
                profile.spans += spans.len() as u64;
            }
            let tick = PROFILE.then(Instant::now);
            grid.group_values(
                &mut [sizes[offset]
                    .get_or_insert_with(|| grid.canvas())
                    .as_mut_slice()],
                board,
                game.position().stones(),
                members,
                [super::schema45_count_feature(members.len())],
            );
            if let Some(tick) = tick {
                profile.painting.record(tick);
                profile.raster.group_shared_shapes.record(tick);
            }
            if let Some(geometry) = &mut geometry {
                geometry.groups.push(GroupGeometry {
                    friendly: offset == 0,
                    members: members.to_vec(),
                    spans: spans.clone(),
                });
            }
            liberties[offset].extend(spans);
        }
    }
    let tick = PROFILE.then(Instant::now);
    let mut stones: Vec<_> = game.position().stones().iter().collect();
    stones.sort_unstable_by_key(|stone| (stone.spot.x, stone.spot.y));
    for stone in stones {
        let offset = usize::from(stone.color != game.on_move());
        raster::paint(
            &mut extractor.coarse[offset * area..(offset + 1) * area],
            board.stone_radius(),
            stone.spot,
            &edges,
        );
    }
    for (offset, spans) in liberties.iter_mut().enumerate() {
        let Some(canvas) = &sizes[offset] else {
            continue;
        };
        let spans = spatial::union_spans(std::mem::take(spans));
        for (plane, values) in [
            (2 + offset, grid.reduce(canvas)),
            (4 + offset, spatial::centers(&spans, &edges)),
            (6 + offset, grid.footprints(&spans, board.stone_radius())),
        ] {
            extractor.coarse[plane * area..(plane + 1) * area].copy_from_slice(&values);
        }
    }
    for y in 0..n {
        for x in 0..n {
            let spot = FeatureExtractor::grid_spot(n, x, y);
            for (direction, value) in extractor.borders(spot).into_iter().enumerate() {
                extractor.coarse[(8 + direction) * area + y * n + x] = value;
            }
        }
    }
    if let Some(tick) = tick {
        profile.painting.record(tick);
    }
    extractor.geometry = geometry.map(std::sync::Arc::new);
    Ok((extractor, profile))
}
