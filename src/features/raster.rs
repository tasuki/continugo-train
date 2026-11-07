//! Analytic area coverage of physical stone discs over raster control volumes.
//! Unlike point sampling/supersampling, partial coverage changes continuously as
//! a stone moves. Same clipped midpoint pixel bounds as ownership targets.
use continugo_rules::{COORD_RANGE, Spot};

/// Area of a unit-disc intersection with [0,x] × [0,y], signed by the quadrant.
fn integral(x: f64, y: f64) -> f64 {
    let sign = x.signum() * y.signum();
    let x = x.abs().min(1.0);
    let y = y.abs().min(1.0);
    if x * x + y * y <= 1.0 {
        return sign * x * y;
    }
    let flat = x.min((1.0 - y * y).max(0.0).sqrt());
    let primitive = |t: f64| 0.5 * (t * (1.0 - t * t).max(0.0).sqrt() + t.asin());
    sign * (flat * y + primitive(x) - primitive(flat))
}

fn coverage(centre: Spot, radius: i32, bounds: [f64; 4]) -> f32 {
    let [left, right, top, bottom] = bounds;
    if right <= left || bottom <= top {
        return 0.0;
    }
    let r = f64::from(radius);
    let x0 = (left - f64::from(centre.x)) / r;
    let x1 = (right - f64::from(centre.x)) / r;
    let y0 = (top - f64::from(centre.y)) / r;
    let y1 = (bottom - f64::from(centre.y)) / r;
    let nearest = |low: f64, high: f64| low.max(0.0).max(-high);
    if nearest(x0, x1).powi(2) + nearest(y0, y1).powi(2) >= 1.0 {
        return 0.0;
    }
    if x0.abs().max(x1.abs()).powi(2) + y0.abs().max(y1.abs()).powi(2) <= 1.0 {
        return 1.0;
    }
    let area = integral(x1, y1) - integral(x0, y1) - integral(x1, y0) + integral(x0, y0);
    (area / ((x1 - x0) * (y1 - y0))).clamp(0.0, 1.0) as f32
}

pub(super) fn paint(plane: &mut [f32], radius: i32, centre: Spot, edges: &[f64]) {
    let resolution = edges.len() - 1;
    let scale = (resolution - 1) as f64 / f64::from(COORD_RANGE - 1);
    // Include neighboring pixels whose area intersects the disc, even when their
    // sample center is outside. Padding covers rounded raster sample locations.
    let bounds = |c: i32| {
        let start = ((f64::from(c - radius) * scale).floor() as i32 - 1).max(0) as usize;
        let end = ((f64::from(c + radius) * scale).ceil() as usize + 1).min(resolution - 1);
        start..=end
    };
    for y in bounds(centre.y) {
        for x in bounds(centre.x) {
            let alpha = coverage(
                centre,
                radius,
                [edges[x], edges[x + 1], edges[y], edges[y + 1]],
            );
            // Legal physical discs have disjoint interiors. Sum their coverage:
            // max() would lose area when two discs intersect the same pixel.
            let pixel = &mut plane[y * resolution + x];
            *pixel = (*pixel + alpha).min(1.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytic_quadrants_and_clipped_pixels() {
        let c = Spot { x: 0, y: 0 };
        let quarter = std::f64::consts::PI as f32 / 4.0;
        for (x, y) in [(0.0, 0.0), (-1.0, 0.0), (0.0, -1.0), (-1.0, -1.0)] {
            assert!((coverage(c, 1, [x, x + 1.0, y, y + 1.0]) - quarter).abs() < 1e-6);
        }
        assert_eq!(coverage(c, 1, [-0.1, 0.1, -0.1, 0.1]), 1.0);
        assert_eq!(coverage(c, 1, [1.0, 2.0, 0.0, 1.0]), 0.0);
        assert_eq!(coverage(c, 1, [0.0, 0.0, 0.0, 1.0]), 0.0);
        assert!((coverage(c, 1, [-1.0, 1.0, -1.0, 1.0]) - quarter).abs() < 1e-6);
    }

    #[test]
    fn coverage_matches_dense_supersampling() {
        let centre = Spot { x: 0, y: 0 };
        for bounds in [
            [-0.8, 0.9, 0.4, 1.2],
            [-1.2, -0.1, -0.9, 0.7],
            [0.1, 0.7, 0.5, 0.9],
        ] {
            let mut inside = 0;
            let n = 512;
            for j in 0..n {
                let y = bounds[2] + (j as f64 + 0.5) / n as f64 * (bounds[3] - bounds[2]);
                for i in 0..n {
                    let x = bounds[0] + (i as f64 + 0.5) / n as f64 * (bounds[1] - bounds[0]);
                    inside += usize::from(x * x + y * y < 1.0);
                }
            }
            let expected = inside as f64 / (n * n) as f64;
            assert!((f64::from(coverage(centre, 1, bounds)) - expected).abs() < 0.0002);
        }
    }
}
