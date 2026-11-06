//! Per-model host context cache. Never retain million-row move features/logits.
//! Exact Game equality includes board, stones, turn and the complete record,
//! so this remains conservative if history-dependent rules are added later.

use continugo_rules::Game;
use std::{collections::VecDeque, sync::Arc};

pub(super) struct Context {
    pub resolution: usize,
    pub spatial: Vec<f32>,
    pub cells: Vec<f32>,
    pub pass: f32,
    pub value: f64,
    pub score: f64,
}

pub(super) struct ContextCache {
    capacity: usize,
    byte_limit: usize,
    pub bytes: usize,
    entries: VecDeque<(Game, Arc<Context>)>,
}

impl ContextCache {
    pub fn new(capacity: usize, byte_limit: usize) -> Self {
        Self {
            capacity,
            byte_limit,
            bytes: 0,
            entries: VecDeque::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn get(&mut self, game: &Game) -> Option<Arc<Context>> {
        let index = self.entries.iter().position(|(key, _)| key == game)?;
        let entry = self.entries.remove(index).unwrap();
        let context = Arc::clone(&entry.1);
        self.entries.push_back(entry);
        Some(context)
    }

    /// Return eviction count. The byte budget covers spatial/cell payloads; keys and
    /// scalar/allocator overhead are additionally bounded by the entry limit.
    pub fn insert(&mut self, game: &Game, context: Arc<Context>) -> usize {
        let bytes = (context.spatial.len() + context.cells.len()) * size_of::<f32>();
        if self.capacity == 0 || bytes > self.byte_limit {
            return 0;
        }
        let mut evictions = 0;
        while self.entries.len() >= self.capacity || self.bytes + bytes > self.byte_limit {
            let (_, oldest) = self.entries.pop_front().unwrap();
            self.bytes -= (oldest.spatial.len() + oldest.cells.len()) * size_of::<f32>();
            evictions += 1;
        }
        self.bytes += bytes;
        self.entries.push_back((game.clone(), context));
        evictions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use continugo_rules::{Board, Spot};

    fn context() -> Arc<Context> {
        Arc::new(Context {
            resolution: 2,
            spatial: vec![0.0; 4],
            cells: Vec::new(),
            pass: 0.0,
            value: 0.25,
            score: -0.1,
        })
    }

    #[test]
    fn lru_is_bounded_and_keys_use_the_whole_game() {
        let a = Game::new(Board::new(9).unwrap());
        let b = Game::new(Board::new(13).unwrap());
        let c = a.play(Spot { x: 500, y: 500 }).unwrap();
        let mut cache = ContextCache::new(2, 1024);
        cache.insert(&a, context());
        cache.insert(&b, context());
        assert!(cache.get(&a).is_some()); // Touch A: B must be evicted next.
        assert_eq!(cache.insert(&c, context()), 1);
        assert!(cache.get(&b).is_none());
        assert!(cache.get(&a).is_some());
        assert!(cache.get(&c).is_some());
        assert!(cache.get(&a.pass().unwrap()).is_none());
        assert!(cache.get(&a.pass().unwrap().pass().unwrap()).is_none());
        assert_eq!(cache.bytes, 32);
        let mut tiny = ContextCache::new(10, 16);
        tiny.insert(&a, context());
        assert_eq!(tiny.insert(&b, context()), 1);
        assert_eq!(tiny.bytes, 16);
        assert_eq!(tiny.len(), 1);
        let mut off = ContextCache::new(0, 1024);
        off.insert(&a, context());
        assert!(off.get(&a).is_none());
        let mut too_small = ContextCache::new(10, 15);
        too_small.insert(&a, context());
        assert_eq!(too_small.len(), 0);
    }
}
