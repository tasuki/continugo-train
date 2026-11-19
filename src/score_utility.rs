//! Static, zero-centered score utility. Scores/stdev are normalized board area.
//! Scale 2 / board_size corresponds to two sqrt(area) point-equivalents.

pub fn score_utility(mean: f64, stdev: f64, board_size: f64) -> f64 {
    let scale = 2.0 / board_size;
    if stdev == 0.0 {
        return std::f64::consts::FRAC_2_PI * (mean / scale).atan();
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_odd_monotone_and_uncertainty_aware() {
        for size in [9.0, 13.0, 19.0] {
            for sigma in [0.0, 0.001, 0.1, 0.5, 1.0] {
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
        for mean in [-0.9, -0.1, 0.0, 0.025, 0.8] {
            for sigma in [0.001, 0.1, 0.5, 1.0] {
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
