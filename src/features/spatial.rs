//! Spatial group/liberty maps for finalized schema 5. Rules provide membership,
//! adjacency and exact integer liberty spans; this module only rasterizes them.
use continugo_rules::{Board, LibertySpan, Spot, Stone};

/// Normal training grids use 8×8 midpoint samples per pixel for unions of discs
/// and bridges/capsules. Bound the intermediate canvas on unusually dense grids.
const SUBSAMPLES: usize = 8;
const MAX_SAMPLE_AXIS: usize = 1024;
const BRIDGE_RADIUS_FRACTION: f64 = 0.30;

pub(super) struct Grid<'a> {
    edges: &'a [f64],
    resolution: usize,
    samples: usize,
    axis: Vec<f64>,
}

impl<'a> Grid<'a> {
    pub fn new(edges: &'a [f64]) -> Self {
        let resolution = edges.len() - 1;
        let samples = SUBSAMPLES.min((MAX_SAMPLE_AXIS / resolution).max(1));
        let axis = edges
            .windows(2)
            .flat_map(|b| {
                (0..samples).map(move |i| b[0] + (b[1] - b[0]) * (i as f64 + 0.5) / samples as f64)
            })
            .collect();
        Self {
            edges,
            resolution,
            samples,
            axis,
        }
    }

    pub fn canvas(&self) -> Vec<f32> {
        vec![0.0; self.axis.len().pow(2)]
    }

    fn indices(&self, low: f64, high: f64) -> std::ops::Range<usize> {
        self.axis.partition_point(|&p| p < low)..self.axis.partition_point(|&p| p <= high)
    }

    fn disc(&self, canvas: &mut [f32], centre: Spot, radius: f64, value: f32) {
        let (cx, cy) = (f64::from(centre.x), f64::from(centre.y));
        let xs = self.indices(cx - radius, cx + radius);
        for y in self.indices(cy - radius, cy + radius) {
            for x in xs.clone() {
                if (self.axis[x] - cx).powi(2) + (self.axis[y] - cy).powi(2) <= radius * radius {
                    let pixel = &mut canvas[y * self.axis.len() + x];
                    *pixel = pixel.max(value);
                }
            }
        }
    }

    fn bridge(&self, canvas: &mut [f32], a: Spot, b: Spot, radius: f64, value: f32) {
        let (a, b) = if (a.x, a.y) <= (b.x, b.y) {
            (a, b)
        } else {
            (b, a)
        };
        let (ax, ay) = (f64::from(a.x), f64::from(a.y));
        let (bx, by) = (f64::from(b.x), f64::from(b.y));
        let (dx, dy) = (bx - ax, by - ay);
        let squared = dx * dx + dy * dy;
        let xs = self.indices(ax.min(bx) - radius, ax.max(bx) + radius);
        for y in self.indices(ay.min(by) - radius, ay.max(by) + radius) {
            for x in xs.clone() {
                let (px, py) = (self.axis[x] - ax, self.axis[y] - ay);
                let t = ((px * dx + py * dy) / squared).clamp(0.0, 1.0);
                if (px - t * dx).powi(2) + (py - t * dy).powi(2) <= radius * radius {
                    let pixel = &mut canvas[y * self.axis.len() + x];
                    *pixel = pixel.max(value);
                }
            }
        }
    }

    pub fn group(
        &self,
        canvas: &mut [f32],
        board: Board,
        stones: &[Stone],
        members: &[usize],
        value: f32,
    ) {
        let radius = f64::from(board.stone_radius());
        for (i, &a) in members.iter().enumerate() {
            self.disc(canvas, stones[a].spot, radius, value);
            for &b in &members[i + 1..] {
                if board.adjacent(stones[a].spot, stones[b].spot) {
                    self.bridge(
                        canvas,
                        stones[a].spot,
                        stones[b].spot,
                        radius * BRIDGE_RADIUS_FRACTION,
                        value,
                    );
                }
            }
        }
    }

    pub fn reduce(&self, canvas: &[f32]) -> Vec<f32> {
        let n = self.resolution;
        let mut result = vec![0.0; n * n];
        for y in 0..n {
            for x in 0..n {
                if self.edges[x + 1] <= self.edges[x] || self.edges[y + 1] <= self.edges[y] {
                    continue;
                }
                let mut sum = 0.0f64;
                for sy in y * self.samples..(y + 1) * self.samples {
                    for sx in x * self.samples..(x + 1) * self.samples {
                        sum += f64::from(canvas[sy * self.axis.len() + sx]);
                    }
                }
                result[y * n + x] = (sum / self.samples.pow(2) as f64) as f32;
            }
        }
        result
    }

    /// Union of stone-radius capsules swept over exact source row spans.
    pub fn footprints(&self, spans: &[LibertySpan], radius: i32) -> Vec<f32> {
        if spans.is_empty() {
            return vec![0.0; self.resolution.pow(2)];
        }
        let radius = f64::from(radius);
        // Accumulate covered sample counts directly into coarse pixels, instead
        // of allocating and then reducing a full supersampled footprint canvas.
        let n = self.resolution;
        let mut counts = vec![0u32; n * n];
        let mut intervals = Vec::new();
        for (y, &coordinate) in self.axis.iter().enumerate() {
            let first = spans.partition_point(|s| f64::from(s.y) < coordinate - radius);
            let last = spans.partition_point(|s| f64::from(s.y) <= coordinate + radius);
            intervals.clear();
            let mut source_row = None;
            let mut extension = 0.0;
            for span in &spans[first..last] {
                if source_row != Some(span.y) {
                    extension = (radius * radius - (f64::from(span.y) - coordinate).powi(2))
                        .max(0.0)
                        .sqrt();
                    source_row = Some(span.y);
                }
                insert_interval(
                    &mut intervals,
                    f64::from(span.left) - extension,
                    f64::from(span.right) + extension,
                );
            }
            let row = &mut counts[y / self.samples * n..(y / self.samples + 1) * n];
            for &(left, right) in &intervals {
                let range = self.indices(left, right);
                if range.is_empty() {
                    continue;
                }
                for (x, count) in row
                    .iter_mut()
                    .enumerate()
                    .take((range.end - 1) / self.samples + 1)
                    .skip(range.start / self.samples)
                {
                    *count += (range.end.min((x + 1) * self.samples)
                        - range.start.max(x * self.samples)) as u32;
                }
            }
        }
        counts
            .into_iter()
            .enumerate()
            .map(|(i, count)| {
                let (x, y) = (i % n, i / n);
                if self.edges[x + 1] <= self.edges[x] || self.edges[y + 1] <= self.edges[y] {
                    0.0
                } else {
                    (f64::from(count) / self.samples.pow(2) as f64) as f32
                }
            })
            .collect()
    }
}

/// Keep only the small disjoint union as source rows arrive. Most neighbouring
/// capsules overlap or are contained; sorting every source interval is wasteful.
fn insert_interval(intervals: &mut Vec<(f64, f64)>, mut left: f64, mut right: f64) {
    let first = intervals.partition_point(|&(_, end)| end < left);
    if first < intervals.len() && intervals[first].0 <= left && right <= intervals[first].1 {
        return;
    }
    let mut last = first;
    while last < intervals.len() && intervals[last].0 <= right {
        left = left.min(intervals[last].0);
        right = right.max(intervals[last].1);
        last += 1;
    }
    if first == last {
        intervals.insert(first, (left, right));
    } else {
        intervals[first] = (left, right);
        intervals.drain(first + 1..last);
    }
}

/// Shared liberties are unioned per colour, not counted once per adjacent group.
pub(super) fn union_spans(mut spans: Vec<LibertySpan>) -> Vec<LibertySpan> {
    spans.sort_unstable_by_key(|s| (s.y, s.left, s.right));
    let mut merged: Vec<LibertySpan> = Vec::new();
    for span in spans {
        if let Some(last) = merged.last_mut()
            && last.y == span.y
            && span.left <= last.right + 1
        {
            last.right = last.right.max(span.right);
            continue;
        }
        merged.push(span);
    }
    merged
}

/// Exact coverage of unit squares about integer liberty-center coordinates.
pub(super) fn centers(spans: &[LibertySpan], edges: &[f64]) -> Vec<f32> {
    let n = edges.len() - 1;
    let mut result = vec![0.0f64; n * n];
    let indices = |low: f64, high: f64| {
        edges
            .partition_point(|&p| p <= low)
            .saturating_sub(1)
            .min(n)..edges.partition_point(|&p| p < high).min(n)
    };
    for span in spans {
        let (left, right) = (f64::from(span.left) - 0.5, f64::from(span.right) + 0.5);
        let (top, bottom) = (f64::from(span.y) - 0.5, f64::from(span.y) + 0.5);
        let xs = indices(left, right);
        for y in indices(top, bottom) {
            for x in xs.clone() {
                let width = edges[x + 1] - edges[x];
                let height = edges[y + 1] - edges[y];
                if width > 0.0 && height > 0.0 {
                    let dx = (right.min(edges[x + 1]) - left.max(edges[x])).max(0.0);
                    let dy = (bottom.min(edges[y + 1]) - top.max(edges[y])).max(0.0);
                    result[y * n + x] += dx * dy / (width * height);
                }
            }
        }
    }
    result
        .into_iter()
        .map(|v| v.clamp(0.0, 1.0) as f32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::targets::raster_bounds;
    use continugo_rules::Game;

    fn integral(values: &[f32], edges: &[f64]) -> f64 {
        let n = edges.len() - 1;
        (0..n)
            .flat_map(|y| {
                (0..n).map(move |x| {
                    f64::from(values[y * n + x])
                        * (edges[x + 1] - edges[x])
                        * (edges[y + 1] - edges[y])
                })
            })
            .sum()
    }

    #[test]
    fn shared_centers_are_unioned_and_area_preserved() {
        let spans = union_spans(vec![
            LibertySpan {
                y: 200,
                left: 200,
                right: 210,
            },
            LibertySpan {
                y: 200,
                left: 205,
                right: 215,
            },
            LibertySpan {
                y: 200,
                left: 200,
                right: 210,
            },
        ]);
        assert_eq!(spans.len(), 1);
        for resolution in [38, 54, 78, 2048] {
            let edges = raster_bounds(resolution);
            let values = centers(&spans, &edges);
            assert!(
                values
                    .iter()
                    .all(|&v| v.is_finite() && (0.0..=1.0).contains(&v))
            );
            assert!((integral(&values, &edges) - 16.0).abs() < 1e-5);
        }
    }

    #[test]
    fn nearby_alternatives_are_not_independent_stone_footprints() {
        let edges = raster_bounds(38);
        let grid = Grid::new(&edges);
        let single = vec![LibertySpan {
            y: 500,
            left: 500,
            right: 500,
        }];
        let nearby = union_spans(
            (500..510)
                .map(|y| LibertySpan {
                    y,
                    left: 500,
                    right: 509,
                })
                .collect(),
        );
        let a = integral(&grid.footprints(&single, 55), &edges);
        let b = integral(&grid.footprints(&nearby, 55), &edges);
        assert!((a / (std::f64::consts::PI * 55.0f64.powi(2)) - 1.0).abs() < 0.01);
        assert!(b > a && b < 1.5 * a);
        assert!((integral(&centers(&single, &edges), &edges) - 1.0).abs() < 1e-6);
        assert!((integral(&centers(&nearby, &edges), &edges) - 100.0).abs() < 1e-5);
        assert!(grid.footprints(&[], 55).iter().all(|&v| v == 0.0));
    }

    #[test]
    fn streamed_footprints_match_canvas_bits_at_all_grid_densities() {
        let spans = union_spans(
            [200, 201, 230, 500, 501, 800]
                .into_iter()
                .flat_map(|y| {
                    [(200, 210), (230, 250), (600, 600)].map(|(left, right)| LibertySpan {
                        y,
                        left,
                        right,
                    })
                })
                .collect(),
        );
        for resolution in [38, 54, 78, 129, 2048] {
            let edges = raster_bounds(resolution);
            let grid = Grid::new(&edges);
            let radius = 55.0f64;
            let mut canvas = grid.canvas();
            for (y, &coordinate) in grid.axis.iter().enumerate() {
                for span in &spans {
                    let dy = f64::from(span.y) - coordinate;
                    if dy.abs() > radius {
                        continue;
                    }
                    let extension = (radius * radius - dy * dy).max(0.0).sqrt();
                    let xs = grid.indices(
                        f64::from(span.left) - extension,
                        f64::from(span.right) + extension,
                    );
                    canvas[y * grid.axis.len() + xs.start..y * grid.axis.len() + xs.end].fill(1.0);
                }
            }
            assert_eq!(grid.footprints(&spans, radius as i32), grid.reduce(&canvas));
        }
    }

    #[test]
    fn only_rules_connected_stones_get_bridges() {
        let edges = raster_bounds(38);
        let grid = Grid::new(&edges);
        for (second, expected) in [(537, true), (600, false)] {
            let game = Game::new(Board::new(9).unwrap())
                .play(Spot { x: 400, y: 500 })
                .unwrap()
                .pass()
                .unwrap()
                .play(Spot { x: second, y: 500 })
                .unwrap();
            let mut canvas = grid.canvas();
            for members in game.position().groups(continugo_rules::Color::Black) {
                grid.group(
                    &mut canvas,
                    game.board(),
                    game.position().stones(),
                    &members,
                    0.2,
                );
            }
            let nearest = |v: f64| {
                (0..grid.axis.len())
                    .min_by(|&a, &b| {
                        (grid.axis[a] - v)
                            .abs()
                            .total_cmp(&(grid.axis[b] - v).abs())
                    })
                    .unwrap()
            };
            let x = nearest((400.0 + f64::from(second)) / 2.0);
            let y = nearest(500.0);
            assert_eq!(canvas[y * grid.axis.len() + x] > 0.0, expected);
            assert!(
                grid.reduce(&canvas)
                    .iter()
                    .all(|&v| (0.0..=0.2).contains(&v))
            );
        }
        assert_eq!(Grid::new(&raster_bounds(2048)).samples, 1);
    }
}
