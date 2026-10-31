//! Area of the UI's actual liberty highlight, in physical stone-disc units.
//!
//! Each exact integer liberty span is a horizontal segment swept by a radius-r
//! disk (a round-ended SVG stroke). Union all these capsules, never sum their
//! overlapping areas. Integrate continuous horizontal union lengths numerically;
//! this is NOT a count of centers, packing capacity, or moves to capture.
//!
//! Midpoint slices are at most min(1 board unit, r/32) high. Legality and liberty
//! centers stay exact; only the displayed continuous area is numerical. Tests
//! compare analytic capsule/disk unions and doubled sampling density. This is
//! tested numerical accuracy, not a certified worst-case integration error bound.

use continugo_rules::LibertySpan;
use std::f64::consts::PI;

pub(crate) fn in_stone_discs(spans: &[LibertySpan], radius: i32) -> f64 {
    measure(spans, f64::from(radius), 1)
}

fn measure(spans: &[LibertySpan], radius: f64, refinement: usize) -> f64 {
    assert!(radius > 0.0);
    let Some(first) = spans.first() else {
        return 0.0;
    };
    debug_assert!(spans.windows(2).all(|pair| pair[0].y <= pair[1].y));
    debug_assert!(spans.iter().all(|s| s.left <= s.right));
    if spans.len() == 1 {
        // Includes an isolated touching center: exactly one stone-disc, not zero.
        return 1.0 + 2.0 * f64::from(first.right - first.left) / (PI * radius);
    }
    let low = f64::from(first.y) - radius;
    let high = f64::from(spans.last().unwrap().y) + radius;
    // Capsule tops/bottoms are integer y coordinates. Align slices to those
    // boundaries so long horizontal spans cannot introduce quadrature jumps.
    let slices_per_unit = (32.0 / radius).ceil().max(1.0) as usize * refinement;
    let slices = (high - low) as usize * slices_per_unit;
    let step = 1.0 / slices_per_unit as f64;
    let mut intervals = Vec::new();
    let mut sum = 0.0;
    for i in 0..slices {
        let y = low + (i as f64 + 0.5) * step;
        let start = spans.partition_point(|s| f64::from(s.y) < y - radius);
        let end = spans.partition_point(|s| f64::from(s.y) <= y + radius);
        intervals.clear();
        for span in &spans[start..end] {
            let dy = y - f64::from(span.y);
            let extension = (radius * radius - dy * dy).max(0.0).sqrt();
            intervals.push((
                f64::from(span.left) - extension,
                f64::from(span.right) + extension,
            ));
        }
        intervals.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        let mut current: Option<(f64, f64)> = None;
        for &(left, right) in &intervals {
            current = match current {
                Some((start, end)) if left <= end => Some((start, end.max(right))),
                Some((start, end)) => {
                    sum += end - start;
                    Some((left, right))
                }
                None => Some((left, right)),
            };
        }
        if let Some((start, end)) = current {
            sum += end - start;
        }
    }
    sum * step / (PI * radius * radius)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_isolated_centers_and_long_capsules_have_analytic_area() {
        for radius in [1, 26, 38, 55] {
            assert_eq!(in_stone_discs(&[], radius), 0.0);
            let point = LibertySpan {
                y: 200,
                left: 200,
                right: 200,
            };
            assert_eq!(in_stone_discs(&[point], radius), 1.0);
            let line = LibertySpan {
                right: 200 + 10 * radius,
                ..point
            };
            let area = in_stone_discs(&[line], radius);
            assert!((area - (1.0 + 20.0 / PI)).abs() < 1e-12);
            assert!(area > 4.0); // No arbitrary cap or clipping to [0,1].
        }
    }

    #[test]
    fn overlapping_and_disjoint_disks_are_unioned_not_double_counted() {
        let radius = 38;
        let point = LibertySpan {
            y: 200,
            left: 200,
            right: 200,
        };
        let other = |dx| LibertySpan {
            left: 200 + dx,
            right: 200 + dx,
            ..point
        };
        assert!((in_stone_discs(&[point, point], radius) - 1.0).abs() < 0.001);
        assert!((in_stone_discs(&[point, other(3 * radius)], radius) - 2.0).abs() < 0.002);
        let expected = 4.0 / 3.0 + 3.0f64.sqrt() / (2.0 * PI);
        assert!((in_stone_discs(&[point, other(radius)], radius) - expected).abs() < 0.002);
    }

    #[test]
    fn long_spans_on_different_rows_preserve_rectangle_area() {
        let spans = [
            LibertySpan {
                y: 100,
                left: 100,
                right: 900,
            },
            LibertySpan {
                y: 300,
                left: 100,
                right: 900,
            },
        ];
        for r in [26, 38, 55] {
            let expected = 2.0 * (1.0 + 1600.0 / (PI * f64::from(r)));
            assert!((in_stone_discs(&spans, r) - expected).abs() < 0.002);
        }
    }

    #[test]
    #[ignore = "release accuracy audit; set CONTINUGO_BENCH_SGF and optionally CONTINUGO_BENCH_PLY (one-based turn)"]
    fn audit_record_areas() {
        use std::{env, fs, time::Instant};
        let path = env::var("CONTINUGO_BENCH_SGF").expect("set CONTINUGO_BENCH_SGF");
        let turn: usize = env::var("CONTINUGO_BENCH_PLY")
            .unwrap_or_else(|_| "1".into())
            .parse()
            .unwrap();
        let ply = turn
            .checked_sub(1)
            .expect("CONTINUGO_BENCH_PLY is one-based");
        let sample = crate::replay::load_position(&fs::read_to_string(path).unwrap(), ply).unwrap();
        let position = sample.game.position();
        let radius = position.board().stone_radius();
        let mut groups = 0;
        let mut max_absolute: f64 = 0.0;
        let mut max_relative: f64 = 0.0;
        let mut base_seconds = 0.0;
        for color in [continugo_rules::Color::Black, continugo_rules::Color::White] {
            for members in position.groups(color) {
                let spans = position.liberty_spans(&members);
                let started = Instant::now();
                let area = in_stone_discs(&spans, radius);
                base_seconds += started.elapsed().as_secs_f64();
                let finer = measure(&spans, f64::from(radius), 2);
                let difference = (area - finer).abs();
                max_absolute = max_absolute.max(difference);
                max_relative = max_relative.max(difference / finer.max(1.0));
                assert!(
                    difference <= 0.003f64.max(0.001 * finer),
                    "{area} vs {finer}"
                );
                groups += 1;
            }
        }
        println!(
            "size={}, ply={ply}, groups={groups}, max_abs={max_absolute:.8} stone-discs, max_relative={max_relative:.8}, area_time={base_seconds:.6}s",
            position.board().size()
        );
    }

    #[test]
    fn real_group_areas_converge_under_doubled_sampling_and_translation() {
        use continugo_rules::{Board, Color, Position, Spot, Stone};
        for size in [9, 13, 19] {
            let board = Board::new(size).unwrap();
            let d = board.diameter();
            let mut position = Position::new(board);
            for (color, x, y) in [
                (Color::Black, 400, 400),
                (Color::Black, 400 + d, 400),
                (Color::White, 400, 400 + d),
                (Color::White, board.stone_radius(), board.stone_radius()),
            ] {
                position = position
                    .play(Stone {
                        color,
                        spot: Spot { x, y },
                    })
                    .unwrap();
            }
            for color in [Color::Black, Color::White] {
                for members in position.groups(color) {
                    let spans = position.liberty_spans(&members);
                    let area = in_stone_discs(&spans, board.stone_radius());
                    let finer = measure(&spans, f64::from(board.stone_radius()), 2);
                    assert!(
                        (area - finer).abs() < 0.003,
                        "size={size}: {area} vs {finer}"
                    );
                    let shifted: Vec<_> = spans
                        .iter()
                        .map(|s| LibertySpan {
                            y: s.y + 11,
                            left: s.left + 17,
                            right: s.right + 17,
                        })
                        .collect();
                    assert!((area - in_stone_discs(&shifted, board.stone_radius())).abs() < 1e-10);
                }
            }
        }
    }
}
