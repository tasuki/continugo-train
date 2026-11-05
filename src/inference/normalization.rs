//! Preserve the global pass/placement split when pruning coarse cells.
//!
//! Selected nonempty cells split the entire placement stratum, not pass mass.
//! Work in log space so tiny selected-cell or placement mass does not underflow.

pub(super) fn cell_log_probs(cells: &[f32], pass: f32, nonempty: &[bool]) -> (Vec<f64>, f64) {
    assert_eq!(cells.len(), nonempty.len());
    let mut probabilities = vec![f64::NEG_INFINITY; cells.len()];
    let selected_max = cells
        .iter()
        .zip(nonempty)
        .filter(|(_, legal)| **legal)
        .map(|(&score, _)| f64::from(score))
        .fold(f64::NEG_INFINITY, f64::max);
    if !selected_max.is_finite() {
        // No selected legal placement. This is not a full-board legality claim.
        return (probabilities, 0.0);
    }
    let cell_max = cells
        .iter()
        .map(|&x| f64::from(x))
        .fold(f64::NEG_INFINITY, f64::max);
    let cell_sum: f64 = cells.iter().map(|&x| (f64::from(x) - cell_max).exp()).sum();
    let maximum = cell_max.max(f64::from(pass));
    let log_denominator =
        ((cell_max - maximum).exp() * cell_sum + (f64::from(pass) - maximum).exp()).ln();
    // Compute both strata directly, rather than subtracting rounded P(pass)
    // from one. Placements keep finite log mass even when pass rounds to 1.
    let placement_logp = cell_max - maximum + cell_sum.ln() - log_denominator;
    let pass_logp = f64::from(pass) - maximum - log_denominator;
    let selected_logsum = cells
        .iter()
        .zip(nonempty)
        .filter(|(_, legal)| **legal)
        .map(|(&score, _)| (f64::from(score) - selected_max).exp())
        .sum::<f64>()
        .ln();
    for ((probability, &score), &legal) in probabilities.iter_mut().zip(cells).zip(nonempty) {
        if legal {
            *probability = placement_logp + f64::from(score) - selected_max - selected_logsum;
        }
    }
    (probabilities, pass_logp)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }

    #[test]
    fn pruning_or_dropping_illegal_cells_never_amplifies_pass() {
        let cells = vec![0.0; 100];
        let pass = (100.0_f32 / 9.0).ln(); // P(pass) = 0.1.
        let expected_pass = f64::from(pass).exp() / (100.0 + f64::from(pass).exp());
        for count in [1, 2, 40, 100] {
            let nonempty: Vec<_> = (0..100).map(|i| i < count).collect();
            let (logp, logpass) = cell_log_probs(&cells, pass, &nonempty);
            close(logpass.exp(), expected_pass);
            close(
                logp.iter().map(|p| p.exp()).sum::<f64>(),
                1.0 - expected_pass,
            );
            close(
                logp.iter().map(|p| p.exp()).sum::<f64>() + logpass.exp(),
                1.0,
            );
            for &p in &logp[..count] {
                close(p.exp(), (1.0 - expected_pass) / count as f64);
            }
            assert!(logp[count..].iter().all(|p| *p == f64::NEG_INFINITY));
        }
    }

    #[test]
    fn conditional_cell_odds_and_extreme_logits_are_stable() {
        let (logp, logpass) = cell_log_probs(&[0.0, 1.0, 1000.0], 0.0, &[true, true, false]);
        assert!(logpass.is_finite());
        close(logp[1] - logp[0], 1.0);
        close(
            logp.iter().map(|p| p.exp()).sum::<f64>() + logpass.exp(),
            1.0,
        );
        let (logp, logpass) = cell_log_probs(&[-1000.0, -999.0], 1000.0, &[true, false]);
        assert!(logp[0].is_finite());
        assert!(logpass.is_finite());
        assert!(logp[0] < -1900.0);
        close(logpass.exp(), 1.0);
    }

    #[test]
    fn no_selected_legal_placement_is_pass_only() {
        let (logp, logpass) = cell_log_probs(&[0.0, 1000.0], -1000.0, &[false, false]);
        assert!(logp.iter().all(|p| *p == f64::NEG_INFINITY));
        assert_eq!(logpass, 0.0);
    }
}
