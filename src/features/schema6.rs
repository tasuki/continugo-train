//! Schema 6 input contract: capped counts/areas and fourteen coarse channels.

pub const VERSION: u32 = 6;
pub const COARSE_PLANES: [&str; 14] = [
    "friendly_stones",
    "enemy_stones",
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
pub const MOVE_FEATURES: [&str; 9] = [
    "connected_groups",
    "connected_stones",
    "captured_groups",
    "captured_stones",
    "splits_friendly_region",
    "left_border",
    "right_border",
    "top_border",
    "bottom_border",
];

/// Log-scaled stone counts and liberty areas saturate at 100 stone-discs.
pub fn scalar_feature(value: f64) -> f32 {
    (value.clamp(0.0, 100.0).ln_1p() / 101.0_f64.ln()) as f32
}

pub fn count_feature(count: usize) -> f32 {
    scalar_feature(count as f64)
}
