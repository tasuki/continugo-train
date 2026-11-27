//! Root-only useless move penalty. All geometry here is raster-sized, never a
//! million-coordinate scan or a tactical/liberty calculation.
use continugo_rules::Game;

pub(super) struct UselessMovePenalty {
    pub resolution: usize,
    pub points: Vec<f32>,
    // Root-local linear conversion from board points to blended search utility.
    pub utility_per_point: f64,
}

impl UselessMovePenalty {
    pub fn new(
        game: &Game,
        ownership: &[f32],
        maximum: f64,
        utility_per_point: f64,
    ) -> Result<Self, String> {
        let n = ownership.len().isqrt();
        if n < 2
            || n * n != ownership.len()
            || ownership
                .iter()
                .any(|v| !v.is_finite() || !(-1.0..=1.0).contains(v))
        {
            return Err("invalid root ownership raster".into());
        }
        // Four-connected, strictly >95% friendly regions. Ownership is STM-relative.
        let mut labels = vec![usize::MAX; ownership.len()];
        let mut stack = Vec::new();
        let mut regions = 0;
        for start in 0..ownership.len() {
            if ownership[start] <= 0.95 || labels[start] != usize::MAX {
                continue;
            }
            labels[start] = regions;
            stack.push(start);
            while let Some(i) = stack.pop() {
                let x = i % n;
                let y = i / n;
                for next in [
                    (x > 0).then(|| i - 1),
                    (x + 1 < n).then_some(i + 1),
                    (y > 0).then(|| i - n),
                    (y + 1 < n).then_some(i + n),
                ]
                .into_iter()
                .flatten()
                {
                    if ownership[next] > 0.95 && labels[next] == usize::MAX {
                        labels[next] = regions;
                        stack.push(next);
                    }
                }
            }
            regions += 1;
        }
        let mut cleanup = vec![false; regions];
        let edges = crate::targets::raster_bounds(n);
        let r = f64::from(game.board().stone_radius());
        let scale = (n - 1) as f64 / 999.0;
        for stone in game
            .position()
            .stones()
            .iter()
            .filter(|s| s.color != game.on_move())
        {
            let cx = f64::from(stone.spot.x);
            let cy = f64::from(stone.spot.y);
            let bounds = |c: f64| {
                let lo = (((c - r) * scale).floor() as isize - 1).max(0) as usize;
                let hi = (((c + r) * scale).ceil() as usize + 1).min(n - 1);
                lo..=hi
            };
            for y in bounds(cy) {
                let dy = (edges[y] - cy).max(0.0).max(cy - edges[y + 1]);
                for x in bounds(cx) {
                    let label = labels[y * n + x];
                    if label == usize::MAX || cleanup[label] {
                        continue;
                    }
                    let dx = (edges[x] - cx).max(0.0).max(cx - edges[x + 1]);
                    // Positive-area disc/pixel overlap; no analytic area integral.
                    if dx * dx + dy * dy < r * r {
                        cleanup[label] = true;
                    }
                }
            }
        }
        let points = ownership
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if v > 0.95 && cleanup[labels[i]] {
                    0.0
                } else {
                    (maximum * ((f64::from(v.abs()) - 0.95) / 0.05).clamp(0.0, 1.0)) as f32
                }
            })
            .collect();
        Ok(Self {
            resolution: n,
            points,
            utility_per_point,
        })
    }

    pub fn utility(&self, action: crate::action::Action) -> f64 {
        if action == crate::action::Action::PASS {
            return 0.0;
        }
        f64::from(self.points[crate::cells::cell(action.index(), self.resolution)])
            * self.utility_per_point
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use continugo_rules::{Action, Board, Spot};

    #[test]
    fn linear_threshold_both_colours_and_pass() {
        let game = Game::default();
        let p = UselessMovePenalty::new(&game, &[0.0, 0.95, -0.975, -1.0], 0.5, 0.02).unwrap();
        assert_eq!(p.points[0], 0.0);
        assert_eq!(p.points[1], 0.0);
        assert!((p.points[2] - 0.25).abs() < 1e-6);
        assert_eq!(p.points[3], 0.5);
        assert_eq!(p.utility(Action::PASS), 0.0);
        assert!((p.utility(Action::from_index(999_999).unwrap()) - 0.01).abs() < 1e-9);
        assert!(UselessMovePenalty::new(&game, &[f32::NAN; 4], 0.5, 1.0).is_err());
        assert!(UselessMovePenalty::new(&game, &[1.0; 5], 0.5, 1.0).is_err());
    }

    #[test]
    fn enemy_overlap_exempts_whole_friendly_region_not_enemy_or_separate_region() {
        // A black stone, now White to play. Black is the enemy.
        let game = Game::new(Board::new(9).unwrap())
            .play(Spot { x: 200, y: 500 })
            .unwrap();
        let n = 54;
        let mut owned = vec![1.0; n * n];
        for y in 0..n {
            owned[y * n + n / 2] = -1.0;
        }
        let p = UselessMovePenalty::new(&game, &owned, 0.5, 1.0).unwrap();
        assert_eq!(p.points[0], 0.0); // Far away, but same region as invader.
        assert_eq!(p.points[n - 1], 0.5); // Separate friendly region.
        assert_eq!(p.points[n / 2], 0.5); // Enemy territory never exempt.
        // With the stone friendly instead, neither region is exempt.
        let friendly = game.pass().unwrap();
        let p = UselessMovePenalty::new(&friendly, &owned, 0.5, 1.0).unwrap();
        assert_eq!(p.points[0], 0.5);
    }

    #[test]
    fn footprint_not_just_stone_centre_and_four_connected_regions() {
        let game = Game::new(Board::new(9).unwrap())
            .play(Spot { x: 200, y: 500 })
            .unwrap();
        let n = 54;
        let mut owned = vec![0.0; n * n];
        let centre = crate::cells::cell(500_200, n);
        let side = centre + 2; // In the physical disc; centre itself is uncertain.
        owned[side] = 1.0;
        owned[side + n + 1] = 1.0; // Diagonal alone does not connect regions.
        let p = UselessMovePenalty::new(&game, &owned, 0.5, 1.0).unwrap();
        assert_eq!(p.points[side], 0.0);
        // Put the isolated region far from the footprint instead.
        owned[n * n - 1] = 1.0;
        let p = UselessMovePenalty::new(&game, &owned, 0.5, 1.0).unwrap();
        assert_eq!(p.points[n * n - 1], 0.5);
    }

    #[test]
    #[ignore = "explicit release microbenchmark"]
    fn benchmark_root_masks() {
        use std::{hint::black_box, time::Instant};
        for size in [9, 13, 19] {
            let mut game = Game::new(Board::new(size).unwrap());
            for y in (100..950).step_by(100) {
                for x in (100..950).step_by(100) {
                    if let Ok(next) = game.play(Spot { x, y }) {
                        game = next;
                    }
                }
            }
            let n = (size * 6) as usize;
            for fragmented in [false, true] {
                let owned: Vec<_> = (0..n * n)
                    .map(|i| {
                        if fragmented && ((i % n).is_multiple_of(7) || (i / n).is_multiple_of(7)) {
                            -1.0
                        } else {
                            1.0
                        }
                    })
                    .collect();
                let start = Instant::now();
                for _ in 0..2000 {
                    black_box(
                        UselessMovePenalty::new(black_box(&game), black_box(&owned), 0.5, 0.01)
                            .unwrap(),
                    );
                }
                eprintln!(
                    "size={size} stones={} raster={n} fragmented={fragmented}: {:.1} us/root",
                    game.position().stones().len(),
                    start.elapsed().as_secs_f64() * 1e6 / 2000.0
                );
            }
        }
    }
}
