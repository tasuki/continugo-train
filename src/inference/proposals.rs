//! Deterministic, position-keyed whole-board exploration, independent of batching.
use continugo_rules::Game;
use sha2::{Digest, Sha256};

pub(super) fn select(
    game: &Game,
    scores: &[f32],
    top: usize,
    exploration: usize,
    query_seed: u64,
) -> Vec<usize> {
    let mut ranked: Vec<_> = (0..scores.len()).collect();
    ranked.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
    let count = top.min(ranked.len());
    let mut selected = ranked[..count].to_vec();
    let mut hash = Sha256::new();
    hash.update(game.board().diameter().to_le_bytes());
    hash.update(query_seed.to_le_bytes());
    for stone in game.position().stones() {
        hash.update(stone.spot.x.to_le_bytes());
        hash.update(stone.spot.y.to_le_bytes());
        hash.update([u8::from(stone.color == game.on_move())]);
    }
    let digest = hash.finalize();
    let seed = u64::from_le_bytes(digest[..8].try_into().unwrap());
    fn mix(mut x: u64) -> u64 {
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
    ranked[count..].sort_unstable_by_key(|&cell| (mix(seed ^ cell as u64), cell));
    selected.extend(ranked[count..].iter().take(exploration));
    selected.sort_unstable();
    selected
}

#[cfg(test)]
mod tests {
    use super::*;
    use continugo_rules::Board;
    #[test]
    fn top_cells_and_distinct_whole_board_exploration_are_stable() {
        let game = Game::new(Board::new(9).unwrap());
        let scores: Vec<_> = (0..100).map(|i| i as f32).collect();
        let a = select(&game, &scores, 4, 3, 0);
        assert_eq!(a, select(&game, &scores, 4, 3, 0));
        assert_ne!(a, select(&game, &scores, 4, 3, 1));
        let mut covered = std::collections::HashSet::new();
        for seed in 0..1000 {
            covered.extend(select(&game, &scores, 4, 3, seed));
        }
        assert_eq!(covered.len(), 100);
        assert_eq!(a.len(), 7);
        assert!(a.windows(2).all(|w| w[0] < w[1]));
        assert!(a.contains(&96) && a.contains(&99));
        assert_eq!(select(&game, &scores, 1000, 8, 0).len(), 100);
    }
}
