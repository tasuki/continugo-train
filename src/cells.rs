//! Non-overlapping nearest-sample cells on the coarse raster. No game rules.
use continugo_rules::{Action, COORD_RANGE};

pub fn cell(action: u32, resolution: usize) -> usize {
    let axis = |x: usize| (x * (resolution - 1) + 499) / 999;
    axis(action as usize % 1000) + resolution * axis(action as usize / 1000)
}

/// Each selected cell contributes every integer coordinate, exactly once.
/// Sorting cells also makes target extraction and inference deterministic.
pub fn actions(cells: &[usize], resolution: usize) -> Vec<Action> {
    assert!(resolution >= 2);
    let mut selected = cells.to_vec();
    selected.sort_unstable();
    selected.dedup();
    let mut axes = vec![Vec::new(); resolution];
    for x in 0..COORD_RANGE {
        axes[(x as usize * (resolution - 1) + 499) / 999].push(x);
    }
    let mut result = Vec::new();
    for cell in selected {
        assert!(cell < resolution * resolution);
        for &y in &axes[cell / resolution] {
            for &x in &axes[cell % resolution] {
                result.push(Action::from_index((x + 1000 * y) as u32).unwrap());
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn partition_covers_every_coordinate_once_including_edges() {
        for resolution in [2, 38, 54, 78] {
            let mut ids: Vec<_> = actions(
                &(0..resolution * resolution).collect::<Vec<_>>(),
                resolution,
            )
            .iter()
            .map(|a| a.index())
            .collect();
            ids.sort_unstable();
            assert_eq!(ids, (0..1_000_000).collect::<Vec<_>>());
            assert_eq!(cell(0, resolution), 0);
            assert_eq!(cell(999_999, resolution), resolution * resolution - 1);
            for c in [0, resolution + 1, resolution * resolution - 1] {
                let selected = actions(&[c, c], resolution);
                assert!(!selected.is_empty());
                assert!(selected.iter().all(|a| cell(a.index(), resolution) == c));
            }
        }
    }
}
