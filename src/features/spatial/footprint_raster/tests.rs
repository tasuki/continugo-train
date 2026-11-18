use super::*;
use crate::{features::spatial::union_spans, targets::raster_bounds};
use rand::{Rng, SeedableRng};

#[test]
fn immutable_kernels_match_scalar_bits_and_reject_other_axes() {
    let mut storage_bytes = 0;
    for (radius, n) in [(55, 38), (38, 54), (26, 78)] {
        let edges = raster_bounds(n);
        let grid = Grid::new(&edges);
        let table = kernel(&grid, radius).unwrap();
        storage_bytes += table.values.capacity() * std::mem::size_of::<f64>()
            + table.axis.capacity() * std::mem::size_of::<f64>()
            + table.rows.capacity() * std::mem::size_of::<KernelRow>()
            + std::mem::size_of::<Kernel>();
        for (y, &coordinate) in grid.axis.iter().enumerate() {
            let last = ((coordinate + f64::from(radius)).floor() as i32).min(999);
            let row = &table.rows[y];
            for sy in row.first..=last {
                let expected = (f64::from(radius).powi(2) - (f64::from(sy) - coordinate).powi(2))
                    .max(0.0)
                    .sqrt();
                assert_eq!(
                    table.values[row.offset + (sy - row.first) as usize].to_bits(),
                    expected.to_bits()
                );
            }
        }
        let mut other = edges.clone();
        other[n / 2] += 0.125;
        let altered = Grid::new(&other);
        assert!(kernel(&altered, radius).is_none());
        let spans = vec![LibertySpan {
            y: 500,
            left: 200,
            right: 700,
        }];
        assert_eq!(
            rasterize(&altered, &spans, radius),
            altered.footprints_reference(&spans, radius)
        );
        let exterior = vec![
            LibertySpan {
                y: -1,
                left: 0,
                right: 100,
            },
            LibertySpan {
                y: 1001,
                left: 0,
                right: 100,
            },
        ];
        assert_eq!(
            rasterize(&grid, &exterior, radius),
            grid.footprints_reference(&exterior, radius)
        );
    }
    assert!(
        storage_bytes < 1024 * 1024,
        "immutable tables: {storage_bytes} bytes"
    );
}

#[test]
fn matches_original_on_fragmented_rows_and_grid_boundaries() {
    let mut rng = rand::rngs::StdRng::seed_from_u64(885522);
    for trial in 0..128 {
        let n = [3, 8, 38, 54, 78, 129][trial % 6];
        let r = [1, 3, 26, 38, 55, 100, 500][trial % 7];
        let mut spans = Vec::new();
        for _ in 0..trial {
            let left = rng.random_range(0..1000);
            spans.push(LibertySpan {
                y: rng.random_range(0..1000),
                left,
                right: (left + rng.random_range(0..250)).min(999),
            });
        }
        let spans = union_spans(spans);
        let edges = raster_bounds(n);
        let grid = Grid::new(&edges);
        assert_eq!(
            rasterize(&grid, &spans, r),
            grid.footprints_reference(&spans, r),
            "trial={trial},n={n},r={r}"
        );
    }
    for edges in [
        vec![0.0, 16.0],
        vec![0.0, 0.0, 16.0, 16.0],
        raster_bounds(2048),
    ] {
        let grid = Grid::new(&edges);
        for r in [1, 26, 55] {
            let spans = union_spans(vec![
                LibertySpan {
                    y: 0,
                    left: 1,
                    right: 1,
                },
                LibertySpan {
                    y: 2,
                    left: 3,
                    right: 3,
                },
                LibertySpan {
                    y: 2,
                    left: 7,
                    right: 11,
                },
                LibertySpan {
                    y: 13,
                    left: 1,
                    right: 15,
                },
                LibertySpan {
                    y: 500,
                    left: 200,
                    right: 700,
                },
            ]);
            assert_eq!(
                rasterize(&grid, &spans, r),
                grid.footprints_reference(&spans, r),
                "r={r}"
            );
        }
    }
}

#[test]
#[ignore = "scratch paired real-corpus footprint benchmark"]
fn profile_real_footprints() {
    use std::{env, fs, hint::black_box, time::Instant};
    let cases: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(env::var("FOOTPRINT_CASES").unwrap()).unwrap())
            .unwrap();
    let mut rows = Vec::new();
    for case in cases.as_array().unwrap() {
        let source = fs::read_to_string(case["sgf"].as_str().unwrap()).unwrap();
        for p in case["plies"].as_array().unwrap() {
            let ply = p.as_u64().unwrap() as usize;
            let sample = crate::replay::load_position(&source, ply).unwrap();
            let game = &sample.game;
            let board = game.board();
            let radius = board.stone_radius();
            let n = crate::features::resolution(board.diameter(), 4).unwrap();
            let edges = raster_bounds(n);
            let grid = Grid::new(&edges);
            let prepared = game.position().prepare();
            let spans: Vec<_> = [game.on_move(), game.on_move().other()]
                .into_iter()
                .map(|color| {
                    union_spans(
                        prepared
                            .groups(color)
                            .into_iter()
                            .flat_map(|g| prepared.liberty_spans(g))
                            .collect(),
                    )
                })
                .collect();
            for s in &spans {
                assert_eq!(
                    grid.footprints_reference(s, radius),
                    rasterize(&grid, s, radius)
                );
            }
            let mut pairs = Vec::new();
            for repeat in 0..12 {
                let mut times = [0.0; 2];
                let mut results = Vec::new();
                for step in 0..2 {
                    let mode = (repeat + step) % 2;
                    let tick = Instant::now();
                    let output: Vec<_> = spans
                        .iter()
                        .map(|s| {
                            if mode == 0 {
                                grid.footprints_reference(black_box(s), radius)
                            } else {
                                rasterize(&grid, black_box(s), radius)
                            }
                        })
                        .collect();
                    times[mode] = tick.elapsed().as_secs_f64();
                    results.push(output);
                }
                assert_eq!(results[0], results[1]);
                black_box(results);
                pairs.push(times);
            }
            rows.push(serde_json::json!({"sgf":case["sgf"],"ply":ply,"size":board.size(),"spans":spans.iter().map(Vec::len).sum::<usize>(),"pairs":pairs,"bit_identical":true}));
        }
    }
    fs::write(
        env::var("FOOTPRINT_OUTPUT").unwrap(),
        serde_json::to_string_pretty(&rows).unwrap() + "\n",
    )
    .unwrap();
}
