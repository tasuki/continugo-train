//! Frozen single-channel geometry oracle for shared group rasterization.
use super::*;
use continugo_rules::Color;

fn reference_group(
    grid: &Grid<'_>,
    canvas: &mut [f32],
    board: Board,
    stones: &[Stone],
    members: &[usize],
    value: f32,
) {
    let radius = f64::from(board.stone_radius());
    for (i, &a) in members.iter().enumerate() {
        let centre = stones[a].spot;
        let (cx, cy) = (f64::from(centre.x), f64::from(centre.y));
        let xs = grid.indices(cx - radius, cx + radius);
        for y in grid.indices(cy - radius, cy + radius) {
            for x in xs.clone() {
                if (grid.axis[x] - cx).powi(2) + (grid.axis[y] - cy).powi(2) <= radius * radius {
                    let pixel = &mut canvas[y * grid.axis.len() + x];
                    *pixel = pixel.max(value);
                }
            }
        }
        for &b in &members[i + 1..] {
            if !board.adjacent(stones[a].spot, stones[b].spot) {
                continue;
            }
            let (a, b) = (stones[a].spot, stones[b].spot);
            let (a, b) = if (a.x, a.y) <= (b.x, b.y) {
                (a, b)
            } else {
                (b, a)
            };
            let (ax, ay) = (f64::from(a.x), f64::from(a.y));
            let (bx, by) = (f64::from(b.x), f64::from(b.y));
            let (dx, dy) = (bx - ax, by - ay);
            let squared = dx * dx + dy * dy;
            let radius = radius * BRIDGE_RADIUS_FRACTION;
            let xs = grid.indices(ax.min(bx) - radius, ax.max(bx) + radius);
            for y in grid.indices(ay.min(by) - radius, ay.max(by) + radius) {
                for x in xs.clone() {
                    let (px, py) = (grid.axis[x] - ax, grid.axis[y] - ay);
                    let t = ((px * dx + py * dy) / squared).clamp(0.0, 1.0);
                    if (px - t * dx).powi(2) + (py - t * dy).powi(2) <= radius * radius {
                        let pixel = &mut canvas[y * grid.axis.len() + x];
                        *pixel = pixel.max(value);
                    }
                }
            }
        }
    }
}

#[test]
fn bridges_are_prominent_but_remain_thinner_than_stones() {
    let edges = crate::targets::raster_bounds(38);
    let grid = Grid::new(&edges);
    let board = Board::new(9).unwrap();
    let stones = [400, 510].map(|x| Stone {
        color: Color::Black,
        spot: Spot { x, y: 500 },
    });
    let mut size = grid.canvas();
    let mut area = grid.canvas();
    grid.group_values(
        &mut [&mut size, &mut area],
        board,
        &stones,
        &[0, 1],
        [0.5, 0.7],
    );
    let nearest = |v: f64| {
        grid.axis
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| (*a - v).abs().total_cmp(&(*b - v).abs()))
            .unwrap()
            .0
    };
    let x = nearest(455.0);
    let inside = nearest(520.0) * grid.axis.len() + x;
    let outside = nearest(530.0) * grid.axis.len() + x;
    // Both points lie outside physical discs. The inner point is beyond the
    // old 0.30*r bridge, but inside 0.40*r; the outer point remains outside.
    for index in [inside, outside] {
        let (x, y) = (
            grid.axis[index % grid.axis.len()],
            grid.axis[index / grid.axis.len()],
        );
        assert!(stones.iter().all(|s| (x - f64::from(s.spot.x)).powi(2)
            + (y - f64::from(s.spot.y)).powi(2)
            > f64::from(board.stone_radius()).powi(2)));
    }
    assert_eq!((size[inside], area[inside]), (0.5, 0.7));
    assert_eq!((size[outside], area[outside]), (0.0, 0.0));
}

#[test]
fn shared_group_coverage_matches_independent_single_channel_oracle() {
    for resolution in [38, 54, 78, 129, 2048] {
        let edges = crate::targets::raster_bounds(resolution);
        let grid = Grid::new(&edges);
        let board = Board::new(9).unwrap();
        let stones: Vec<_> = [
            (55, 55),
            (165, 55),
            (400, 500),
            (510, 500),
            (620, 500),
            (550, 800),
        ]
        .into_iter()
        .map(|(x, y)| Stone {
            color: Color::Black,
            spot: Spot { x, y },
        })
        .collect();
        let mut size = grid.canvas();
        let mut area = grid.canvas();
        let mut old_size = grid.canvas();
        let mut old_area = grid.canvas();
        // Overlapping shapes deliberately choose different winners in each
        // channel. Also exercise edges, disconnected members, bridge orientation,
        // zero/one values, and different supersampling densities.
        for (members, values) in [
            (vec![1, 0], [0.7, 0.1]),
            (vec![2, 3, 4], [0.2, 0.9]),
            (vec![4, 3, 2], [0.8, 0.3]),
            (vec![5], [0.0, 1.0]),
            (vec![0, 5], [0.1, 0.2]),
        ] {
            reference_group(&grid, &mut old_size, board, &stones, &members, values[0]);
            reference_group(&grid, &mut old_area, board, &stones, &members, values[1]);
            grid.group_values(
                &mut [&mut size, &mut area],
                board,
                &stones,
                &members,
                values,
            );
            assert_eq!(size, old_size, "size canvas, resolution {resolution}");
            assert_eq!(area, old_area, "area canvas, resolution {resolution}");
        }
        assert_eq!(grid.reduce(&size), grid.reduce(&old_size));
        assert_eq!(grid.reduce(&area), grid.reduce(&old_area));
    }
}
