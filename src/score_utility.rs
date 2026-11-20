//! Static, zero-centered score utility. Scores/stdev are normalized net margins.
//! Scale 2 / SZ corresponds to two-SZ board points, independent of komi.

pub fn score_utility(mean: f64, stdev: f64, board_size: f64) -> f64 {
    let scale = 2.0 / board_size;
    if stdev == 0.0 {
        return std::f64::consts::FRAC_2_PI * (mean / scale).atan();
    }
    if stdev > 1.0 {
        // Large uncertainty makes the arctan transition too narrow for the
        // fixed grid. Refine only where needed; ordinary leaves keep its cache.
        let f = |z: f64| {
            (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
                * (((mean + stdev * z) / scale).atan() + ((mean - stdev * z) / scale).atan())
        };
        let (fa, fm, fb) = (f(0.0), f(4.0), f(8.0));
        let whole = 8.0 / 6.0 * (fa + 4.0 * fm + fb);
        return (std::f64::consts::FRAC_2_PI
            * adaptive_simpson(&f, 0.0, 8.0, [fa, fm, fb], whole, 1e-7, 20))
        .clamp(-1.0, 1.0);
    }
    // Simpson integration of a standard Gaussian on [-8,8], paired to preserve
    // exact odd symmetry. Cache the Gaussian weights, shared by all searches.
    static WEIGHTS: std::sync::OnceLock<[f64; 129]> = std::sync::OnceLock::new();
    let weights = WEIGHTS.get_or_init(|| {
        std::array::from_fn(|i| {
            let z = i as f64 / 16.0;
            let coefficient = if i == 128 {
                1.0
            } else if i % 2 == 1 {
                4.0
            } else {
                2.0
            };
            coefficient * (-0.5 * z * z).exp() / (48.0 * (2.0 * std::f64::consts::PI).sqrt())
        })
    });
    let mut result = weights[0] * (mean / scale).atan();
    for (i, weight) in weights.iter().enumerate().skip(1) {
        let delta = stdev * i as f64 / 16.0;
        result += weight * (((mean + delta) / scale).atan() + ((mean - delta) / scale).atan());
    }
    (std::f64::consts::FRAC_2_PI * result).clamp(-1.0, 1.0)
}

fn adaptive_simpson(
    f: &impl Fn(f64) -> f64,
    a: f64,
    b: f64,
    values: [f64; 3],
    whole: f64,
    tolerance: f64,
    depth: u32,
) -> f64 {
    let [fa, fm, fb] = values;
    let m = (a + b) / 2.0;
    let (fl, fr) = (f((a + m) / 2.0), f((m + b) / 2.0));
    let left = (m - a) / 6.0 * (fa + 4.0 * fl + fm);
    let right = (b - m) / 6.0 * (fm + 4.0 * fr + fb);
    let error = left + right - whole;
    if depth == 0 || error.abs() <= 15.0 * tolerance {
        return left + right + error / 15.0;
    }
    adaptive_simpson(f, a, m, [fa, fl, fm], left, tolerance / 2.0, depth - 1)
        + adaptive_simpson(f, m, b, [fm, fr, fb], right, tolerance / 2.0, depth - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_odd_monotone_and_uncertainty_aware() {
        for size in [9.0, 13.0, 19.0] {
            for sigma in [0.0, 0.001, 0.1, 0.5, 1.0, 2.0, 10.0] {
                let mut previous = -1.0;
                for i in -100..=100 {
                    let mean = i as f64 / 100.0;
                    let u = score_utility(mean, sigma, size);
                    assert!((-1.0..=1.0).contains(&u));
                    assert!(u >= previous);
                    assert!((u + score_utility(-mean, sigma, size)).abs() < 1e-14);
                    previous = u;
                }
            }
            assert!(score_utility(0.2, 0.5, size) < score_utility(0.2, 0.0, size));
        }
    }

    #[test]
    fn quadrature_matches_fine_reference() {
        for mean in [-3.0, -0.9, -0.1, 0.0, 0.025, 0.8, 3.0] {
            for sigma in [0.001, 0.1, 0.5, 1.0, 2.0, 10.0] {
                let step = 16.0 / 32768.0;
                let mut reference = 0.0;
                for i in 0..32768 {
                    let z = -8.0 + (i as f64 + 0.5) * step;
                    reference += score_utility(mean + sigma * z, 0.0, 19.0) * (-0.5 * z * z).exp()
                        / (2.0 * std::f64::consts::PI).sqrt()
                        * step;
                }
                assert!((score_utility(mean, sigma, 19.0) - reference).abs() < 0.0002);
            }
        }
    }
}
