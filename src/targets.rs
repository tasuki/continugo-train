//! Auxiliary labels from validated final Voronoi geometry, never unfinished territory.
use continugo_rules::{
    Color, Stone,
    territory::{self, Cell, Point},
};
use std::sync::{Arc, Mutex, OnceLock};

pub const BOARD_AREA: f64 = 1_000_000.0;
pub const TARGET_VERSION: &str = "final-area-v1";

type Raster = (usize, Arc<Vec<f32>>);

pub struct FinalTargets {
    pub score: f32,
    stones: Vec<Stone>,
    cells: OnceLock<Vec<Cell>>,
    // One density at a time, bounded to the worker's current game.
    raster: Mutex<Option<Raster>>,
}

impl FinalTargets {
    pub fn new(stones: Vec<Stone>, areas: territory::Areas) -> Self {
        Self {
            score: ((areas.black - areas.white) / BOARD_AREA) as f32,
            stones,
            cells: OnceLock::new(),
            raster: Mutex::new(None),
        }
    }

    pub fn ownership(&self, resolution: usize) -> Arc<Vec<f32>> {
        let mut cached = self.raster.lock().expect("target raster lock");
        if let Some((size, raster)) = &*cached
            && *size == resolution
        {
            return Arc::clone(raster);
        }
        let cells = self.cells.get_or_init(|| territory::cells(&self.stones));
        let bounds = raster_bounds(resolution);
        let mut raster = vec![0.0f64; resolution * resolution];
        for cell in cells {
            if cell.polygon.is_empty() {
                continue;
            }
            let min_x = cell
                .polygon
                .iter()
                .map(|p| p.x)
                .fold(f64::INFINITY, f64::min);
            let max_x = cell
                .polygon
                .iter()
                .map(|p| p.x)
                .fold(f64::NEG_INFINITY, f64::max);
            let min_y = cell
                .polygon
                .iter()
                .map(|p| p.y)
                .fold(f64::INFINITY, f64::min);
            let max_y = cell
                .polygon
                .iter()
                .map(|p| p.y)
                .fold(f64::NEG_INFINITY, f64::max);
            let sign = if cell.color == Color::Black {
                1.0
            } else {
                -1.0
            };
            for y in 0..resolution {
                if bounds[y + 1] <= min_y || bounds[y] >= max_y {
                    continue;
                }
                for x in 0..resolution {
                    if bounds[x + 1] <= min_x || bounds[x] >= max_x {
                        continue;
                    }
                    let mut polygon = cell.polygon.clone();
                    for (axis, boundary, keep_greater) in [
                        (0, bounds[x], true),
                        (0, bounds[x + 1], false),
                        (1, bounds[y], true),
                        (1, bounds[y + 1], false),
                    ] {
                        polygon = clip(&polygon, axis, boundary, keep_greater);
                    }
                    let area = (bounds[x + 1] - bounds[x]) * (bounds[y + 1] - bounds[y]);
                    if area > 0.0 {
                        raster[y * resolution + x] += sign * polygon_area(&polygon) / area;
                    }
                }
            }
        }
        let raster = Arc::new(
            raster
                .into_iter()
                .map(|v| v.clamp(-1.0, 1.0) as f32)
                .collect(),
        );
        *cached = Some((resolution, Arc::clone(&raster)));
        raster
    }
}

/// Continuous control volumes about the same rounded sample locations as features.
pub fn raster_bounds(resolution: usize) -> Vec<f64> {
    assert!(resolution >= 2);
    let sample = |i: usize| ((i * 999 + (resolution - 1) / 2) / (resolution - 1)) as f64;
    std::iter::once(0.0)
        .chain((1..resolution).map(|i| (sample(i - 1) + sample(i)) * 0.5))
        .chain(std::iter::once(1000.0))
        .collect()
}

fn clip(polygon: &[Point], axis: usize, boundary: f64, keep_greater: bool) -> Vec<Point> {
    let Some(&last) = polygon.last() else {
        return Vec::new();
    };
    let side = |p: Point| {
        let d = (if axis == 0 { p.x } else { p.y }) - boundary;
        if keep_greater { -d } else { d }
    };
    let mut result = Vec::new();
    let mut previous = last;
    for &current in polygon {
        let a = side(previous);
        let b = side(current);
        if (a <= 0.0) != (b <= 0.0) {
            let t = a / (a - b);
            result.push(Point {
                x: previous.x + t * (current.x - previous.x),
                y: previous.y + t * (current.y - previous.y),
            });
        }
        if b <= 0.0 {
            result.push(current);
        }
        previous = current;
    }
    result
}

fn polygon_area(polygon: &[Point]) -> f64 {
    let Some(&last) = polygon.last() else {
        return 0.0;
    };
    let mut previous = last;
    let mut twice = 0.0;
    for &current in polygon {
        twice += previous.x * current.y - previous.y * current.x;
        previous = current;
    }
    twice.abs() * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use continugo_rules::{Board, Game, Spot};

    #[test]
    fn captured_stones_do_not_own_final_territory() {
        let game = Game::new(Board::new(13).unwrap())
            .play(Spot { x: 38, y: 38 })
            .unwrap()
            .play(Spot { x: 92, y: 92 })
            .unwrap()
            .pass()
            .unwrap()
            .pass()
            .unwrap();
        assert_eq!(game.position().stones().len(), 1);
        let target = FinalTargets::new(
            game.position().stones().to_vec(),
            territory::areas(game.position().stones()),
        );
        assert_eq!(target.score, -1.0);
        assert!(target.ownership(54).iter().all(|&v| v == -1.0));
    }

    #[test]
    fn symmetric_boundary_cells_are_soft_and_color_swap_flips_them() {
        let game = Game::new(Board::new(13).unwrap())
            .play(Spot { x: 250, y: 500 })
            .unwrap()
            .play(Spot { x: 750, y: 500 })
            .unwrap();
        let target = FinalTargets::new(
            game.position().stones().to_vec(),
            territory::areas(game.position().stones()),
        );
        assert_eq!(target.score, 0.0);
        let raster = target.ownership(5);
        for y in 0..5 {
            assert!(raster[y * 5 + 2].abs() < 0.01);
        }
        let swapped: Vec<_> = game
            .position()
            .stones()
            .iter()
            .map(|s| Stone {
                color: s.color.other(),
                spot: s.spot,
            })
            .collect();
        let swap = FinalTargets::new(swapped.clone(), territory::areas(&swapped));
        for (&a, &b) in raster.iter().zip(swap.ownership(5).iter()) {
            assert_eq!(a, -b);
        }
        // Duplicate sample locations at extreme supported densities have no NaNs.
        let dense = target.ownership(1002);
        assert!(dense.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn final_rasters_integrate_to_score_at_every_board_size() {
        for size in [9, 13, 19] {
            let empty = Game::new(Board::new(size).unwrap());
            let black = empty.play(Spot { x: 200, y: 300 }).unwrap();
            let split = black.play(Spot { x: 700, y: 600 }).unwrap();
            for game in [&empty, &black, &split] {
                let stones = game.position().stones();
                let target = FinalTargets::new(stones.to_vec(), territory::areas(stones));
                let n = crate::features::resolution(game.board().diameter(), 4).unwrap();
                let raster = target.ownership(n);
                assert!(Arc::ptr_eq(&raster, &target.ownership(n)));
                let bounds = raster_bounds(n);
                let mut integral = 0.0;
                for y in 0..n {
                    for x in 0..n {
                        let v = raster[y * n + x];
                        assert!((-1.0..=1.0).contains(&v));
                        integral += f64::from(v)
                            * (bounds[x + 1] - bounds[x])
                            * (bounds[y + 1] - bounds[y])
                            / BOARD_AREA;
                    }
                }
                assert!((integral - f64::from(target.score)).abs() < 1e-7);
                if game.record().is_empty() {
                    assert!(raster.iter().all(|&v| v == 0.0));
                }
            }
        }
    }
}
