//! Board-budget scheduling, not game rules. Matches Python sampling.py.
use std::collections::BTreeMap;

pub const DEFAULT_SIZE_WEIGHTS: &str = "9:2,13:2,19:1";

pub fn weighted_sizes(sizes: &[i32], text: &str) -> Result<Vec<i32>, String> {
    if sizes.is_empty() || sizes.iter().any(|s| !(9..=19).contains(s)) {
        return Err("board sizes must be nonempty and in 9..19".into());
    }
    if text == "uniform" {
        return Ok(sizes.to_vec());
    }
    let mut weights = BTreeMap::new();
    for item in text.split(',') {
        let (size, weight) = item
            .split_once(':')
            .ok_or("expected size:weight pairs or uniform")?;
        let size: i32 = size.parse().map_err(|_| "invalid weight size")?;
        let weight: i64 = weight.parse().map_err(|_| "invalid size weight")?;
        if !(9..=19).contains(&size)
            || !(0..=1000).contains(&weight)
            || weights.insert(size, weight).is_some()
        {
            return Err("weights need unique sizes 9..19 and integer weights 0..1000".into());
        }
    }
    let mut entries: Vec<(i32, i64)> = Vec::new();
    for &size in sizes {
        let weight = *weights.get(&size).unwrap_or(&1);
        if let Some(entry) = entries.iter_mut().find(|e| e.0 == size) {
            entry.1 += weight;
        } else if weight > 0 {
            entries.push((size, weight));
        }
    }
    fn gcd(mut a: i64, mut b: i64) -> i64 {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a
    }
    let divisor = entries
        .iter()
        .map(|e| e.1)
        .reduce(gcd)
        .ok_or("no positive-weight board sizes available")?;
    for entry in &mut entries {
        entry.1 /= divisor;
    }
    let total: i64 = entries.iter().map(|e| e.1).sum();
    if total > 4096 {
        return Err("weighted board cycle exceeds 4096 entries".into());
    }
    let mut current = vec![0; entries.len()];
    let mut result = Vec::new();
    for _ in 0..total {
        for (i, entry) in entries.iter().enumerate() {
            current[i] += entry.1;
        }
        let mut best = 0;
        for i in 1..entries.len() {
            if current[i] > current[best] {
                best = i;
            }
        }
        result.push(entries[best].0);
        current[best] -= total;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn default_mix_and_legacy_cycles() {
        assert_eq!(
            weighted_sizes(&[9, 13, 19], DEFAULT_SIZE_WEIGHTS).unwrap(),
            vec![9, 13, 19, 9, 13]
        );
        assert_eq!(
            weighted_sizes(&[13], DEFAULT_SIZE_WEIGHTS).unwrap(),
            vec![13]
        );
        assert_eq!(
            weighted_sizes(&[9, 13, 19], "uniform").unwrap(),
            vec![9, 13, 19]
        );
        assert_eq!(
            weighted_sizes(&[9, 13, 19], "9:2,13:2,19:0").unwrap(),
            vec![9, 13]
        );
        assert!(weighted_sizes(&[9], "9:0").is_err());
        assert!(weighted_sizes(&[9], "9:1,9:2").is_err());
    }
}
