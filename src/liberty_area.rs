//! Area of the UI's actual liberty highlight, in physical stone-disc units.
//!
//! Each exact integer liberty span is a horizontal segment swept by a radius-r
//! disk (a round-ended SVG stroke). Union all these capsules, never sum their
//! overlapping areas. Integrate continuous horizontal union lengths numerically;
//! this is NOT a count of centers, packing capacity, or moves to capture.
//!
//! The precise legacy/diagnostic integral uses midpoint slices at most
//! min(1 board unit, r/32) high. Schema-6 coarse inputs use eight times that
//! spacing, with a clipped final slice. Legality and liberty centers stay exact.
//! Both are numerical estimates, not certified worst-case integration bounds.

use continugo_rules::LibertySpan;
use std::{f64::consts::PI, sync::OnceLock};

/// Preserve the original integral for legacy inputs and accuracy diagnostics.
pub(crate) fn in_stone_discs(spans: &[LibertySpan], radius: i32) -> f64 {
    measure(spans, f64::from(radius), 1)
}

/// Approved schema-6 input precision. Same capsule union/analytic shortcuts,
/// eight times the midpoint spacing; no memoization or feature-layout change.
pub(crate) fn coarse_input_in_stone_discs(spans: &[LibertySpan], radius: i32) -> f64 {
    measure_impl::<false, true>(spans, f64::from(radius), 1)
}

/// Same midpoint samples as legacy extraction. Cache circular extensions only
/// when binary-exact grid spacing guarantees identical floating-point endpoints.
#[cfg(test)]
pub(crate) fn candidate_in_stone_discs(spans: &[LibertySpan], radius: i32) -> f64 {
    let density = (32.0 / f64::from(radius)).ceil().max(1.0) as usize;
    if !(1..=500).contains(&radius) || !density.is_power_of_two() {
        return in_stone_discs(spans, radius);
    }
    measure_impl::<true, false>(spans, f64::from(radius), 1)
}

fn measure(spans: &[LibertySpan], radius: f64, refinement: usize) -> f64 {
    measure_impl::<false, false>(spans, radius, refinement)
}

fn measure_impl<const KERNEL: bool, const COARSE_INPUT: bool>(
    spans: &[LibertySpan],
    radius: f64,
    refinement: usize,
) -> f64 {
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
    // Preserve the precise grid exactly for legacy/diagnostic callers. Coarse
    // inputs retain its lower anchor and cover the last partial slice separately.
    let slices_per_unit = (32.0 / radius).ceil().max(1.0) as usize * refinement;
    let step = if COARSE_INPUT { 8.0 } else { 1.0 } / slices_per_unit as f64;
    let slices = if COARSE_INPUT {
        ((high - low) / step).ceil() as usize
    } else {
        (high - low) as usize * slices_per_unit
    };
    static KERNELS: [OnceLock<Vec<f64>>; 501] = [const { OnceLock::new() }; 501];
    let kernel = if KERNEL {
        KERNELS[radius as usize]
            .get_or_init(|| {
                (0..2 * radius as usize * slices_per_unit)
                    .map(|i| {
                        let dy = -radius + (i as f64 + 0.5) * step;
                        (radius * radius - dy * dy).max(0.0).sqrt()
                    })
                    .collect()
            })
            .as_slice()
    } else {
        &[]
    };
    // Maintain the sorted disjoint union directly. Most slices have only one
    // or two components, even when hundreds of source spans overlap them. This
    // avoids materializing and sorting every capsule interval on every slice.
    let mut intervals: Vec<(f64, f64)> = Vec::new();
    let mut sum = 0.0;
    for i in 0..slices {
        let bottom = low + i as f64 * step;
        let height = if COARSE_INPUT {
            (high - bottom).min(step)
        } else {
            step
        };
        let y = if COARSE_INPUT {
            bottom + height * 0.5
        } else {
            low + (i as f64 + 0.5) * step
        };
        let start = spans.partition_point(|s| f64::from(s.y) < y - radius);
        let end = spans.partition_point(|s| f64::from(s.y) <= y + radius);
        intervals.clear();
        let mut row = None;
        let mut extension = 0.0;
        for span in &spans[start..end] {
            // Disjoint spans on the same source row share the same round end.
            if row != Some(span.y) {
                extension = if KERNEL {
                    let index = i as isize - (span.y - first.y) as isize * slices_per_unit as isize;
                    kernel[index as usize]
                } else {
                    let dy = y - f64::from(span.y);
                    (radius * radius - dy * dy).max(0.0).sqrt()
                };
                row = Some(span.y);
            }
            let mut left = f64::from(span.left) - extension;
            let mut right = f64::from(span.right) + extension;
            // Most slices have one connected union. Updating it directly avoids
            // binary searches and Vec draining for every overlapping source row.
            if intervals.len() == 1 && left <= intervals[0].1 && right >= intervals[0].0 {
                intervals[0].0 = intervals[0].0.min(left);
                intervals[0].1 = intervals[0].1.max(right);
                continue;
            }
            let first = intervals.partition_point(|&(_, end)| end < left);
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
        for &(start, end) in &intervals {
            if COARSE_INPUT {
                sum += (end - start) * height;
            } else {
                sum += end - start;
            }
        }
    }
    if COARSE_INPUT {
        sum / (PI * radius * radius)
    } else {
        sum * step / (PI * radius * radius)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Frozen pre-optimization sort-and-merge implementation. Keep independent
    /// of the streaming union so fragmented/bridging intervals have an oracle.
    fn reference(spans: &[LibertySpan], radius: f64, refinement: usize) -> f64 {
        let Some(first) = spans.first() else {
            return 0.0;
        };
        if spans.len() == 1 {
            return 1.0 + 2.0 * f64::from(first.right - first.left) / (PI * radius);
        }
        let low = f64::from(first.y) - radius;
        let high = f64::from(spans.last().unwrap().y) + radius;
        let slices_per_unit = (32.0 / radius).ceil().max(1.0) as usize * refinement;
        let step = 1.0 / slices_per_unit as f64;
        let mut intervals = Vec::new();
        let mut sum = 0.0;
        for i in 0..(high - low) as usize * slices_per_unit {
            let y = low + (i as f64 + 0.5) * step;
            intervals.clear();
            for span in spans {
                if f64::from(span.y) < y - radius || f64::from(span.y) > y + radius {
                    continue;
                }
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

    #[test]
    fn schema6_coarse_input_matches_approved_eightfold_precision() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/liberty-area8.json")).unwrap();
        assert_eq!(fixture["coarsening"].as_u64(), Some(8));
        for case in fixture["scalar_cases"].as_array().unwrap() {
            let radius = case["radius"].as_i64().unwrap() as i32;
            let spans: Vec<_> = case["spans"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| LibertySpan {
                    y: s[0].as_i64().unwrap() as i32,
                    left: s[1].as_i64().unwrap() as i32,
                    right: s[2].as_i64().unwrap() as i32,
                })
                .collect();
            let precise = in_stone_discs(&spans, radius);
            let coarse = coarse_input_in_stone_discs(&spans, radius);
            assert_eq!(
                precise.to_bits(),
                case["precise_area_bits"].as_u64().unwrap(),
                "precise: {case}"
            );
            assert_eq!(
                coarse.to_bits(),
                case["area8_bits"].as_u64().unwrap(),
                "coarse: {case}"
            );
            assert_eq!(
                crate::features::scalar_feature(coarse).to_bits(),
                case["feature8_bits"].as_u64().unwrap() as u32
            );
            assert!(
                (crate::features::scalar_feature(coarse)
                    - crate::features::scalar_feature(precise))
                .abs()
                    < 0.002
            );
            let shifted: Vec<_> = spans
                .iter()
                .map(|s| LibertySpan {
                    y: s.y + 7,
                    left: s.left + 11,
                    right: s.right + 11,
                })
                .collect();
            assert!((coarse_input_in_stone_discs(&shifted, radius) - coarse).abs() < 1e-10);
        }
    }

    #[test]
    fn streaming_union_matches_sort_reference_bits() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(271828);
        for radius in [1, 3, 7, 26, 38, 55, 100, 500] {
            for trial in 0..32 {
                let mut spans = Vec::new();
                // Duplicate rows, nested intervals, disjoint components and
                // later intervals bridging earlier components in arbitrary x order.
                for _ in 0..trial * 3 {
                    let left = rng.random_range(-100..1000);
                    spans.push(LibertySpan {
                        y: rng.random_range(-20..80),
                        left,
                        right: left + rng.random_range(0..200),
                    });
                }
                spans.sort_by_key(|s| s.y);
                assert_eq!(
                    candidate_in_stone_discs(&spans, radius).to_bits(),
                    in_stone_discs(&spans, radius).to_bits(),
                    "candidate radius={radius}, trial={trial}"
                );
                for refinement in [1, 2] {
                    assert_eq!(
                        measure(&spans, f64::from(radius), refinement).to_bits(),
                        reference(&spans, f64::from(radius), refinement).to_bits(),
                        "radius={radius}, trial={trial}, refinement={refinement}"
                    );
                }
            }
        }
    }

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
                assert_eq!(
                    area.to_bits(),
                    reference(&spans, f64::from(radius), 1).to_bits()
                );
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
                    assert_eq!(
                        area.to_bits(),
                        reference(&spans, f64::from(board.stone_radius()), 1).to_bits()
                    );
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
