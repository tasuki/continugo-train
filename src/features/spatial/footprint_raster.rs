//! Exact original raster samples, with precomputed circle extensions and a
//! direct common-case interval update. No per-group/per-position memoization.
use super::Grid;
use continugo_rules::{COORD_RANGE, LibertySpan};
use std::sync::OnceLock;

struct KernelRow {
    first: i32,
    offset: usize,
}
struct Kernel {
    axis: Vec<f64>,
    rows: Vec<KernelRow>,
    values: Vec<f64>,
}
impl Kernel {
    fn new(axis: &[f64], radius: i32) -> Self {
        let r = f64::from(radius);
        let mut rows = Vec::with_capacity(axis.len());
        let mut values = Vec::new();
        for &y in axis {
            let first = ((y - r).ceil() as i32).max(0);
            let last = ((y + r).floor() as i32).min(COORD_RANGE - 1);
            rows.push(KernelRow {
                first,
                offset: values.len(),
            });
            for sy in first..=last {
                values.push((r * r - (f64::from(sy) - y).powi(2)).max(0.0).sqrt());
            }
        }
        Self {
            axis: axis.to_vec(),
            rows,
            values,
        }
    }
}

/// Three bounded immutable geometry tables for default density-4 model grids.
/// Equality of the actual sample axes is checked, never assumed from dimensions.
fn kernel(grid: &Grid<'_>, radius: i32) -> Option<&'static Kernel> {
    static N9: OnceLock<Kernel> = OnceLock::new();
    static N13: OnceLock<Kernel> = OnceLock::new();
    static N19: OnceLock<Kernel> = OnceLock::new();
    let slot = match (radius, grid.resolution, grid.samples) {
        (55, 38, 8) => &N9,
        (38, 54, 8) => &N13,
        (26, 78, 8) => &N19,
        _ => return None,
    };
    let table = slot.get_or_init(|| {
        let edges = crate::targets::raster_bounds(grid.resolution);
        let canonical = Grid::new(&edges);
        Kernel::new(&canonical.axis, radius)
    });
    (table.axis == grid.axis).then_some(table)
}

#[cfg(test)]
mod tests;

fn insert(intervals: &mut Vec<(f64, f64)>, mut left: f64, mut right: f64) {
    if intervals.len() == 1 && left <= intervals[0].1 && right >= intervals[0].0 {
        intervals[0].0 = intervals[0].0.min(left);
        intervals[0].1 = intervals[0].1.max(right);
        return;
    }
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

pub(super) fn rasterize(grid: &Grid<'_>, spans: &[LibertySpan], radius: i32) -> Vec<f32> {
    let n = grid.resolution;
    if spans.is_empty() {
        return vec![0.0; n * n];
    }
    let table = if spans[0].y >= 0 && spans.last().unwrap().y < COORD_RANGE {
        kernel(grid, radius)
    } else {
        None
    };
    let radius = f64::from(radius);
    let mut counts = vec![0u32; n * n];
    let mut intervals = Vec::new();
    for (y, &coordinate) in grid.axis.iter().enumerate() {
        let first = spans.partition_point(|s| f64::from(s.y) < coordinate - radius);
        let last = spans.partition_point(|s| f64::from(s.y) <= coordinate + radius);
        intervals.clear();
        let mut source_row = None;
        let mut extension = 0.0;
        for span in &spans[first..last] {
            if source_row != Some(span.y) {
                extension = if let Some(table) = table {
                    let row = &table.rows[y];
                    table.values[row.offset + (span.y - row.first) as usize]
                } else {
                    (radius * radius - (f64::from(span.y) - coordinate).powi(2))
                        .max(0.0)
                        .sqrt()
                };
                source_row = Some(span.y);
            }
            insert(
                &mut intervals,
                f64::from(span.left) - extension,
                f64::from(span.right) + extension,
            );
        }
        let row = &mut counts[y / grid.samples * n..(y / grid.samples + 1) * n];
        for &(left, right) in &intervals {
            let range = grid.indices(left, right);
            if range.is_empty() {
                continue;
            }
            for (x, count) in row
                .iter_mut()
                .enumerate()
                .take((range.end - 1) / grid.samples + 1)
                .skip(range.start / grid.samples)
            {
                *count += (range.end.min((x + 1) * grid.samples)
                    - range.start.max(x * grid.samples)) as u32;
            }
        }
    }
    counts
        .into_iter()
        .enumerate()
        .map(|(i, count)| {
            let (x, y) = (i % n, i / n);
            if grid.edges[x + 1] <= grid.edges[x] || grid.edges[y + 1] <= grid.edges[y] {
                0.0
            } else {
                (f64::from(count) / grid.samples.pow(2) as f64) as f32
            }
        })
        .collect()
}
