//! Binary splitting of a friendly-bounded region of available integer centres.
//! This is a geometric input, not a life/legality rule. Eight-neighbour topology
//! retains single-coordinate passages. Captures use the actual child survivors.
use continugo_rules::{Board, Color, Game, Position, Spot, Stone};
use std::{
    cell::RefCell,
    collections::VecDeque,
    sync::{Arc, Mutex, OnceLock},
};

const CAPTURE_STATES: usize = 16;
const CAPTURE_BYTES: usize = 4 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
struct Run {
    left: i32,
    right: i32,
    label: usize,
}
#[derive(Clone, Copy, Debug)]
struct RowRun {
    y: i32,
    left: i32,
    right: i32,
}
#[derive(Default)]
struct Region {
    boundary: u8,
    rows: Vec<RowRun>,
}

struct Index {
    low: i32,
    high: i32,
    widths: Vec<i32>,
    rows: Vec<Vec<Run>>,
    regions: Vec<Region>,
}

fn root(parents: &mut [usize], mut i: usize) -> usize {
    while parents[i] != i {
        parents[i] = parents[parents[i]];
        i = parents[i];
    }
    i
}
fn join(parents: &mut [usize], a: usize, b: usize) -> bool {
    let a = root(parents, a);
    let b = root(parents, b);
    if a == b {
        return false;
    }
    // Stable smaller root keeps repeated scanline joins shallow.
    parents[a.max(b)] = a.min(b);
    true
}
fn merge(row: &mut Vec<(i32, i32)>) {
    row.sort_unstable();
    let mut n = 0;
    for i in 0..row.len() {
        let (left, right) = row[i];
        if n > 0 && left <= row[n - 1].1 + 1 {
            row[n - 1].1 = row[n - 1].1.max(right);
        } else {
            row[n] = (left, right);
            n += 1;
        }
    }
    row.truncate(n);
}
fn touches(row: &[(i32, i32)], left: i32, right: i32) -> bool {
    let first = row.partition_point(|&(_, end)| end < left);
    row.get(first).is_some_and(|&(start, _)| start <= right)
}

impl Index {
    fn new(board: Board, stones: &[Stone]) -> Self {
        let low = board.stone_radius();
        let high = 1000 - low;
        let d = board.diameter();
        let widths: Vec<_> = (0..d).map(|dy| (d * d - 1 - dy * dy).isqrt()).collect();
        let mut blocked: [Vec<Vec<(i32, i32)>>; 2] =
            std::array::from_fn(|_| vec![Vec::new(); 1000]);
        for stone in stones {
            let color = usize::from(stone.color == Color::White);
            for y in low.max(stone.spot.y - d + 1)..=high.min(stone.spot.y + d - 1) {
                let w = widths[(y - stone.spot.y).unsigned_abs() as usize];
                blocked[color][y as usize]
                    .push(((stone.spot.x - w).max(low), (stone.spot.x + w).min(high)));
            }
        }
        for color in &mut blocked {
            for row in color.iter_mut() {
                merge(row);
            }
        }
        let mut rows: Vec<Vec<Run>> = vec![Vec::new(); 1000];
        let mut parents = Vec::new();
        let mut union = Vec::new();
        for y in low..=high {
            union.clear();
            union.extend_from_slice(&blocked[0][y as usize]);
            union.extend_from_slice(&blocked[1][y as usize]);
            merge(&mut union);
            let mut left = low;
            let mut emit = |left, right| {
                let label = parents.len();
                parents.push(label);
                rows[y as usize].push(Run { left, right, label });
            };
            for &(start, end) in &union {
                if left < start {
                    emit(left, start - 1);
                }
                left = left.max(end + 1);
            }
            if left <= high {
                emit(left, high);
            }
            if y > low {
                for run in &rows[y as usize] {
                    let previous = &rows[y as usize - 1];
                    let first = previous.partition_point(|p| p.right + 1 < run.left);
                    for p in &previous[first..] {
                        if p.left > run.right + 1 {
                            break;
                        }
                        join(&mut parents, run.label, p.label);
                    }
                }
            }
        }
        let mut regions: Vec<Region> = Vec::new();
        let mut labels = vec![usize::MAX; parents.len()];
        for y in low..=high {
            for run in &mut rows[y as usize] {
                let parent = root(&mut parents, run.label);
                if labels[parent] == usize::MAX {
                    labels[parent] = regions.len();
                    regions.push(Region::default());
                }
                run.label = labels[parent];
                let region = &mut regions[run.label];
                region.rows.push(RowRun {
                    y,
                    left: run.left,
                    right: run.right,
                });
                // A region's boundary consists of blocked 8-neighbours. Edges
                // of the permitted centre rectangle are neutral eye boundaries.
                for (color, color_rows) in blocked.iter().enumerate() {
                    if region.boundary & (1 << color) != 0 {
                        continue;
                    }
                    if ((y - 1).max(low)..=(y + 1).min(high))
                        .any(|yy| touches(&color_rows[yy as usize], run.left - 1, run.right + 1))
                    {
                        region.boundary |= 1 << color;
                    }
                }
            }
        }
        Self {
            low,
            high,
            widths,
            rows,
            regions,
        }
    }

    fn bytes(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.widths.capacity() * 4
            + self.rows.capacity() * std::mem::size_of::<Vec<Run>>()
            + self
                .rows
                .iter()
                .map(|r| r.capacity() * std::mem::size_of::<Run>())
                .sum::<usize>()
            + self.regions.capacity() * std::mem::size_of::<Region>()
            + self
                .regions
                .iter()
                .map(|r| r.rows.capacity() * std::mem::size_of::<RowRun>())
                .sum::<usize>()
    }

    fn splits(&self, spot: Spot, color: Color) -> bool {
        if spot.y < self.low || spot.y > self.high {
            return false;
        }
        let row = &self.rows[spot.y as usize];
        let end = row.partition_point(|r| r.left <= spot.x);
        let Some(run) = end
            .checked_sub(1)
            .and_then(|i| row.get(i))
            .filter(|r| r.right >= spot.x)
        else {
            return false;
        };
        let region = &self.regions[run.label];
        if region.boundary != 1 << usize::from(color == Color::White) {
            return false;
        }
        SCRATCH.with(|scratch| {
            scratch
                .borrow_mut()
                .splits(&region.rows, spot, &self.widths)
        })
    }
}

#[derive(Default)]
struct Scratch {
    parents: Vec<usize>,
    previous: Vec<Run>,
    current: Vec<Run>,
}
thread_local! { static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default()); }
impl Scratch {
    fn splits(&mut self, rows: &[RowRun], spot: Spot, widths: &[i32]) -> bool {
        self.parents.clear();
        self.previous.clear();
        self.current.clear();
        let mut components = 0usize;
        let mut current_y = i32::MIN;
        for run in rows {
            if run.y != current_y {
                std::mem::swap(&mut self.previous, &mut self.current);
                self.current.clear();
                if run.y != current_y + 1 {
                    self.previous.clear();
                }
                current_y = run.y;
            }
            let dy = (run.y - spot.y).unsigned_abs() as usize;
            let mut emit = |left, right| {
                if left > right {
                    return;
                }
                let label = self.parents.len();
                self.parents.push(label);
                components += 1;
                let first = self.previous.partition_point(|p| p.right + 1 < left);
                for p in &self.previous[first..] {
                    if p.left > right + 1 {
                        break;
                    }
                    if join(&mut self.parents, label, p.label) {
                        components -= 1;
                    }
                }
                self.current.push(Run { left, right, label });
            };
            if let Some(&w) = widths.get(dy) {
                emit(run.left, run.right.min(spot.x - w - 1));
                emit(run.left.max(spot.x + w + 1), run.right);
            } else {
                emit(run.left, run.right);
            }
        }
        components >= 2
    }
}

struct Captured {
    stones: Vec<Stone>,
    index: Arc<Index>,
    bytes: usize,
}
#[derive(Default)]
struct Captures {
    entries: VecDeque<Captured>,
    bytes: usize,
}
#[derive(Default)]
pub(super) struct Cache {
    parent: OnceLock<Arc<Index>>,
    captures: Mutex<Captures>,
}
impl Cache {
    /// Called only for authoritative legal moves. For captures (including any
    /// removed friendly stones), inspect actual survivors with the new stone
    /// temporarily absent. Identical survivor sets share a bounded exact index.
    pub fn value(&self, game: &Game, spot: Spot, child: &Position) -> f32 {
        let old = game.position().stones();
        let survivors = child.stones();
        debug_assert_eq!(survivors.last().map(|s| s.spot), Some(spot));
        let survivors = &survivors[..survivors.len() - 1];
        if survivors.is_empty() {
            return 0.0;
        }
        let index = if survivors.len() == old.len() {
            Arc::clone(
                self.parent
                    .get_or_init(|| Arc::new(Index::new(game.board(), old))),
            )
        } else {
            let found = {
                let cache = self.captures.lock().unwrap();
                cache
                    .entries
                    .iter()
                    .find(|e| e.stones == survivors)
                    .map(|e| Arc::clone(&e.index))
            };
            if let Some(index) = found {
                return f32::from(index.splits(spot, game.on_move()));
            }
            let index = Arc::new(Index::new(game.board(), survivors));
            let bytes = index.bytes() + std::mem::size_of_val(survivors);
            if bytes <= CAPTURE_BYTES {
                let mut cache = self.captures.lock().unwrap();
                if !cache.entries.iter().any(|e| e.stones == survivors) {
                    while cache.bytes + bytes > CAPTURE_BYTES
                        || cache.entries.len() >= CAPTURE_STATES
                    {
                        let entry = cache.entries.pop_front().unwrap();
                        cache.bytes -= entry.bytes;
                    }
                    cache.bytes += bytes;
                    cache.entries.push_back(Captured {
                        stones: survivors.to_vec(),
                        index: Arc::clone(&index),
                        bytes,
                    });
                }
            }
            index
        };
        f32::from(index.splits(spot, game.on_move()))
    }
}

#[cfg(test)]
mod tests;
