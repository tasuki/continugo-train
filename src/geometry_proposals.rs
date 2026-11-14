//! Opt-in direct geometric proposals, using exact group spans from coarse input.
//! No liberty-count scoring, large preliminary pool, or learned-head changes.
use crate::cells;
use continugo_rules::{Action, Board, Game, LibertySpan, Spot};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::Serialize;
use std::{collections::BTreeMap, time::Instant};

pub const VERSION: &str = "geometry-v2-frontier";
pub const DEFAULT_BUDGET: usize = 16;
pub const EXTEND: u8 = 1;
pub const JOIN: u8 = 2;
pub const HUG: u8 = 4;
// Bit 8 was Breathe in v1; v2 has no such generator/scorer.
pub const CAPTURE: u8 = 16;
pub const COVERAGE: u8 = 32;
const DIRECTIONS: [(i32, i32); 8] = [
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
];
const AXES: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

#[derive(Clone, Debug)]
pub struct GroupGeometry {
    pub friendly: bool,
    pub members: Vec<usize>,
    pub spans: Vec<LibertySpan>,
}
#[derive(Clone, Debug, Default)]
pub struct Geometry {
    pub groups: Vec<GroupGeometry>,
}
impl Geometry {
    pub fn bytes(&self) -> usize {
        self.groups.capacity() * size_of::<GroupGeometry>()
            + self
                .groups
                .iter()
                .map(|g| {
                    g.members.capacity() * size_of::<usize>()
                        + g.spans.capacity() * size_of::<LibertySpan>()
                })
                .sum::<usize>()
    }
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
pub struct Rect {
    pub left: i32,
    pub right: i32,
    pub top: i32,
    pub bottom: i32,
}
impl Rect {
    fn contains(self, p: Spot) -> bool {
        (self.left..=self.right).contains(&p.x) && (self.top..=self.bottom).contains(&p.y)
    }
    fn expanded(self, d: i32) -> Self {
        Self {
            left: self.left - d,
            right: self.right + d,
            top: self.top - d,
            bottom: self.bottom + d,
        }
    }
    fn overlaps(self, b: Self) -> bool {
        self.left <= b.right && b.left <= self.right && self.top <= b.bottom && b.top <= self.bottom
    }
    fn center(self) -> Spot {
        Spot {
            x: (self.left + self.right) / 2,
            y: (self.top + self.bottom) / 2,
        }
    }
    fn distance(self, p: Spot) -> i64 {
        if self.contains(p) {
            0
        } else {
            distance(
                p,
                Spot {
                    x: p.x.clamp(self.left, self.right),
                    y: p.y.clamp(self.top, self.bottom),
                },
            )
        }
    }
}
/// Exactly the nearest-sample cell partition, including empty high-density cells.
pub fn cell_rect(cell: usize, resolution: usize) -> Rect {
    assert!(resolution >= 2 && cell < resolution * resolution);
    let bounds = |owner: usize| {
        let owns = |x: &i32| ((*x as usize * (resolution - 1) + 499) / 999) == owner;
        (
            (0..1000).find(owns).unwrap_or(1),
            (0..1000).rfind(owns).unwrap_or(0),
        )
    };
    let (left, right) = bounds(cell % resolution);
    let (top, bottom) = bounds(cell / resolution);
    Rect {
        left,
        right,
        top,
        bottom,
    }
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct Candidate {
    pub action: u32,
    /// Generator provenance, not all effects a random coordinate might achieve.
    pub sources: u8,
    pub protected: bool,
    /// A true connection/board frontier, not a cell edge or overlap boundary.
    pub frontier: bool,
    pub groups: Vec<usize>,
    /// Groups for which this coordinate was explicitly generated as an extension.
    pub extension_groups: Vec<usize>,
}
impl Candidate {
    pub fn spot(&self) -> Spot {
        Spot {
            x: (self.action % 1000) as i32,
            y: (self.action / 1000) as i32,
        }
    }
    pub fn source_names(&self) -> Vec<&'static str> {
        [
            (EXTEND, "extend"),
            (JOIN, "join"),
            (HUG, "hug"),
            (CAPTURE, "capture"),
            (COVERAGE, "random"),
        ]
        .into_iter()
        .filter_map(|(b, n)| (self.sources & b != 0).then_some(n))
        .collect()
    }
}

/// Disjoint clocks. Source counts overlap because coordinates are deduplicated.
#[derive(Debug, Default, Serialize)]
pub struct ProposalProfile {
    pub setup_seconds: f64,
    pub extend_seconds: f64,
    pub join_seconds: f64,
    pub capture_seconds: f64,
    pub random_seconds: f64,
    pub tag_seconds: f64,
    pub finish_seconds: f64,
    /// Extend, Join, Hug, Capture, Random.
    pub final_source_counts: [u64; 5],
}
impl ProposalProfile {
    pub fn merge(&mut self, other: &Self) {
        macro_rules! add { ($($f:ident),*) => { $(self.$f += other.$f;)* }; }
        add!(
            setup_seconds,
            extend_seconds,
            join_seconds,
            capture_seconds,
            random_seconds,
            tag_seconds,
            finish_seconds
        );
        for (a, b) in self
            .final_source_counts
            .iter_mut()
            .zip(other.final_source_counts)
        {
            *a += b;
        }
    }
    pub fn measured_seconds(&self) -> f64 {
        self.setup_seconds
            + self.extend_seconds
            + self.join_seconds
            + self.capture_seconds
            + self.random_seconds
            + self.tag_seconds
            + self.finish_seconds
    }
}
fn elapsed(timer: Option<Instant>, total: &mut f64) {
    if let Some(t) = timer {
        *total += t.elapsed().as_secs_f64();
    }
}
#[derive(Debug, Serialize)]
pub struct Shortlist {
    pub version: &'static str,
    pub selected_cells: Vec<usize>,
    pub candidates: Vec<Candidate>,
    /// v2 sends the direct deduplicated pool to exact analysis without prescoring.
    pub preliminary_candidates: usize,
    pub protected_overflow: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub profile: Option<ProposalProfile>,
}
impl Shortlist {
    pub fn actions(&self) -> Vec<Action> {
        self.candidates
            .iter()
            .map(|c| Action::from_index(c.action).unwrap())
            .collect()
    }
}

fn distance(a: Spot, b: Spot) -> i64 {
    i64::from(a.x - b.x).pow(2) + i64::from(a.y - b.y).pow(2)
}
fn bounds(s: &[LibertySpan]) -> Option<Rect> {
    Some(Rect {
        left: s.iter().map(|s| s.left).min()?,
        right: s.iter().map(|s| s.right).max()?,
        top: s.first()?.y,
        bottom: s.last()?.y,
    })
}
fn clip(s: &[LibertySpan], r: Rect) -> Vec<LibertySpan> {
    let start = s.partition_point(|s| s.y < r.top);
    s[start..]
        .iter()
        .take_while(|s| s.y <= r.bottom)
        .filter_map(|s| {
            let (left, right) = (s.left.max(r.left), s.right.min(r.right));
            (left <= right).then_some(LibertySpan {
                y: s.y,
                left,
                right,
            })
        })
        .collect()
}
fn contains(s: &[LibertySpan], p: Spot) -> bool {
    let start = s.partition_point(|s| s.y < p.y);
    s[start..]
        .iter()
        .take_while(|s| s.y == p.y)
        .any(|s| (s.left..=s.right).contains(&p.x))
}
fn extreme(s: &[LibertySpan], (dx, dy): (i32, i32), origin: Spot) -> Option<Spot> {
    s.iter()
        .map(|s| Spot {
            x: if dx > 0 {
                s.right
            } else if dx < 0 {
                s.left
            } else {
                origin.x.clamp(s.left, s.right)
            },
            y: s.y,
        })
        .max_by_key(|&p| {
            (
                i64::from(dx) * i64::from(p.x) + i64::from(dy) * i64::from(p.y),
                -distance(p, origin),
                -(p.x + 1000 * p.y),
            )
        })
}
fn nearest(s: &[LibertySpan], target: Spot) -> Option<Spot> {
    s.iter()
        .map(|s| Spot {
            x: target.x.clamp(s.left, s.right),
            y: s.y,
        })
        .min_by_key(|&p| (distance(p, target), p.x + 1000 * p.y))
}
fn intersect(a: &[LibertySpan], b: &[LibertySpan]) -> Vec<LibertySpan> {
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::new();
    while i < a.len() && j < b.len() {
        let (x, y) = (a[i], b[j]);
        if x.y < y.y {
            i += 1;
            continue;
        }
        if y.y < x.y {
            j += 1;
            continue;
        }
        let (left, right) = (x.left.max(y.left), x.right.min(y.right));
        if left <= right {
            out.push(LibertySpan {
                y: x.y,
                left,
                right,
            });
        }
        if x.right <= y.right { i += 1 } else { j += 1 }
    }
    out
}
fn union(mut s: Vec<LibertySpan>) -> Vec<LibertySpan> {
    s.sort_unstable_by_key(|s| (s.y, s.left));
    let mut out: Vec<LibertySpan> = Vec::new();
    for s in s {
        if let Some(last) = out.last_mut()
            && last.y == s.y
            && s.left <= last.right + 1
        {
            last.right = last.right.max(s.right);
            continue;
        }
        out.push(s);
    }
    out
}
fn subtract(a: &[LibertySpan], b: &[LibertySpan]) -> Vec<LibertySpan> {
    let mut out = Vec::new();
    for s in a {
        let mut left = s.left;
        let start = b.partition_point(|b| b.y < s.y);
        for t in b[start..]
            .iter()
            .take_while(|t| t.y == s.y && t.left <= s.right)
        {
            if t.right < left {
                continue;
            }
            if t.left > left {
                out.push(LibertySpan {
                    y: s.y,
                    left,
                    right: t.left - 1,
                });
            }
            left = left.max(t.right + 1);
            if left > s.right {
                break;
            }
        }
        if left <= s.right {
            out.push(LibertySpan {
                y: s.y,
                left,
                right: s.right,
            });
        }
    }
    out
}
/// Eight-neighbour integer components without enumerating their points.
fn components(s: &[LibertySpan]) -> Vec<Vec<LibertySpan>> {
    fn root(p: &mut [usize], mut i: usize) -> usize {
        while p[i] != i {
            p[i] = p[p[i]];
            i = p[i];
        }
        i
    }
    let mut parent: Vec<_> = (0..s.len()).collect();
    for (i, a) in s.iter().enumerate() {
        let start = s[..i].partition_point(|b| b.y < a.y - 1);
        for (j, b) in s.iter().enumerate().take(i).skip(start) {
            if b.y == a.y - 1 && b.left <= a.right + 1 && a.left <= b.right + 1 {
                let x = root(&mut parent, i);
                let y = root(&mut parent, j);
                parent[x] = y;
            }
        }
    }
    let mut out: BTreeMap<usize, Vec<LibertySpan>> = BTreeMap::new();
    for (i, &s) in s.iter().enumerate() {
        out.entry(root(&mut parent, i)).or_default().push(s);
    }
    out.into_values().collect()
}
/// Exact regions connecting each group set, including tiny three-way overlaps.
fn join_regions(
    local: &[Vec<LibertySpan>],
    friendly: &[usize],
) -> BTreeMap<Vec<usize>, Vec<LibertySpan>> {
    let mut rows: BTreeMap<i32, Vec<(i32, usize, bool)>> = BTreeMap::new();
    for &i in friendly {
        for s in &local[i] {
            let e = rows.entry(s.y).or_default();
            e.push((s.left, i, true));
            e.push((s.right + 1, i, false));
        }
    }
    let mut out: BTreeMap<Vec<usize>, Vec<LibertySpan>> = BTreeMap::new();
    for (y, mut e) in rows {
        e.sort_unstable();
        let mut active = std::collections::BTreeSet::new();
        let mut left = e[0].0;
        for (x, i, enters) in e {
            if x > left && active.len() >= 2 {
                out.entry(active.iter().copied().collect())
                    .or_default()
                    .push(LibertySpan {
                        y,
                        left,
                        right: x - 1,
                    });
            }
            if enters {
                active.insert(i);
            } else {
                active.remove(&i);
            }
            left = x;
        }
    }
    out
}
fn width(sq: i64, dy: i32) -> Option<i32> {
    let r = sq - i64::from(dy).pow(2);
    (r >= 0).then(|| (r as u64).isqrt() as i32)
}

/// Shared lazy reach rows distinguish the real connection limit from cell edges
/// and overlap holes. This is proposal geometry, never an authoritative validator.
struct ReachRows<'a> {
    game: &'a Game,
    geometry: &'a Geometry,
    rows: BTreeMap<(usize, i32), Vec<LibertySpan>>,
}
impl<'a> ReachRows<'a> {
    fn new(game: &'a Game, geometry: &'a Geometry) -> Self {
        Self {
            game,
            geometry,
            rows: BTreeMap::new(),
        }
    }
    fn row(&mut self, group: usize, y: i32) -> &[LibertySpan] {
        self.rows.entry((group, y)).or_insert_with(|| {
            let board = self.game.board();
            let (low, high) = (board.stone_radius(), 1000 - board.stone_radius());
            if !(low..=high).contains(&y) {
                return Vec::new();
            }
            let sq = 2 * i64::from(board.diameter()).pow(2) - 1;
            union(
                self.geometry.groups[group]
                    .members
                    .iter()
                    .filter_map(|&i| {
                        let p = self.game.position().stones()[i].spot;
                        width(sq, y - p.y).and_then(|w| {
                            let (left, right) = ((p.x - w).max(low), (p.x + w).min(high));
                            (left <= right).then_some(LibertySpan { y, left, right })
                        })
                    })
                    .collect(),
            )
        })
    }
    fn outer(&mut self, group: usize, local: &[LibertySpan]) -> Vec<LibertySpan> {
        let mut interior = Vec::new();
        let mut previous = None;
        for s in local {
            if previous == Some(s.y) {
                continue;
            }
            previous = Some(s.y);
            let mut eroded: Option<Vec<LibertySpan>> = None;
            for y in s.y - 1..=s.y + 1 {
                let row: Vec<_> = self
                    .row(group, y)
                    .iter()
                    .filter_map(|r| {
                        (r.left < r.right - 1).then_some(LibertySpan {
                            y: s.y,
                            left: r.left + 1,
                            right: r.right - 1,
                        })
                    })
                    .collect();
                eroded = Some(match eroded {
                    None => row,
                    Some(e) => intersect(&e, &row),
                });
            }
            interior.extend(eroded.unwrap());
        }
        subtract(local, &interior)
    }
}
/// Exact farthest-from-nearest-friendly point over spans. On each row the nearest
/// site changes only at pairwise bisectors; between them squared distance is
/// convex, so endpoints and the two integers around each bisector suffice.
fn farthest(spans: &[LibertySpan], sites: &[Spot]) -> Option<Spot> {
    let mut best: Option<(i64, i32, Spot)> = None;
    for s in spans {
        let mut consider = |x: i32| {
            if x < s.left || x > s.right {
                return;
            }
            let p = Spot { x, y: s.y };
            let Some(reach) = sites.iter().map(|&q| distance(p, q)).min() else {
                return;
            };
            let key = (reach, -(x + 1000 * s.y));
            if best.as_ref().is_none_or(|b| key > (b.0, b.1)) {
                best = Some((key.0, key.1, p));
            }
        };
        consider(s.left);
        consider(s.right);
        for (i, a) in sites.iter().enumerate() {
            for b in &sites[..i] {
                let mut denominator = 2 * i64::from(b.x - a.x);
                if denominator == 0 {
                    continue;
                }
                let mut numerator = i64::from(b.x).pow(2) + i64::from(b.y).pow(2)
                    - i64::from(a.x).pow(2)
                    - i64::from(a.y).pow(2)
                    - 2 * i64::from(s.y) * i64::from(b.y - a.y);
                if denominator < 0 {
                    denominator = -denominator;
                    numerator = -numerator;
                }
                let x = numerator.div_euclid(denominator) as i32;
                consider(x);
                consider(x + 1);
            }
        }
    }
    best.map(|b| b.2)
}
/// Free connected centers within one integer-neighbour step of enemy overlap.
/// Includes tangencies even if hugging lies inside our maximum reach boundary.
fn contact_band(target: Spot, rect: Rect, d: i32) -> Vec<LibertySpan> {
    let sq = i64::from(d).pow(2) - 1;
    (rect.top..=rect.bottom)
        .filter_map(|y| {
            let w = (-1..=1)
                .filter_map(|dy| width(sq, y + dy - target.y))
                .max()?
                + 1;
            let (left, right) = (
                (target.x - w).max(rect.left),
                (target.x + w).min(rect.right),
            );
            (left <= right).then_some(LibertySpan { y, left, right })
        })
        .collect()
}
/// Two-group lenses are oriented by their local supporting stones, not compass
/// axes. The first two extrema are the two side tips perpendicular to that axis.
fn join_directions(
    game: &Game,
    geometry: &Geometry,
    joined: &[usize],
    region: &[LibertySpan],
) -> [(i32, i32); 4] {
    if joined.len() != 2 {
        return AXES;
    }
    let anchor = nearest(region, bounds(region).unwrap().center()).unwrap();
    let support = |i: usize| {
        geometry.groups[i]
            .members
            .iter()
            .map(|&k| game.position().stones()[k].spot)
            .min_by_key(|&p| (distance(p, anchor), p.x, p.y))
    };
    let (Some(a), Some(b)) = (support(joined[0]), support(joined[1])) else {
        return AXES;
    };
    let (dx, dy) = (b.x - a.x, b.y - a.y);
    if (dx, dy) == (0, 0) {
        return AXES;
    }
    [(-dy, dx), (dy, -dx), (dx, dy), (-dx, -dy)]
}

/// Placement geometry only; actual suicide/capture validation happens afterward.
struct FreeRows<'a> {
    game: &'a Game,
    rows: Vec<Option<Vec<LibertySpan>>>,
}
impl<'a> FreeRows<'a> {
    fn new(game: &'a Game) -> Self {
        Self {
            game,
            rows: vec![None; 1000],
        }
    }
    fn region(&mut self, rect: Rect) -> Vec<LibertySpan> {
        let board = self.game.board();
        let (low, high) = (board.stone_radius(), 1000 - board.stone_radius());
        let mut out = Vec::new();
        for y in rect.top.max(low)..=rect.bottom.min(high) {
            let row = self.rows[y as usize].get_or_insert_with(|| {
                let sq = i64::from(board.diameter()).pow(2) - 1;
                let mut blocked: Vec<_> = self
                    .game
                    .position()
                    .stones()
                    .iter()
                    .filter_map(|s| width(sq, y - s.spot.y).map(|w| (s.spot.x - w, s.spot.x + w)))
                    .collect();
                blocked.sort_unstable();
                let mut start = low;
                let mut free = Vec::new();
                for (a, b) in blocked {
                    if start < a && start <= high {
                        free.push(LibertySpan {
                            y,
                            left: start,
                            right: (a - 1).min(high),
                        });
                    }
                    start = start.max(b + 1);
                }
                if start <= high {
                    free.push(LibertySpan {
                        y,
                        left: start,
                        right: high,
                    });
                }
                free
            });
            out.extend(row.iter().filter_map(|s| {
                let (left, right) = (s.left.max(rect.left), s.right.min(rect.right));
                (left <= right).then_some(LibertySpan { y, left, right })
            }));
        }
        out
    }
}
/// Cover the complete enemy liberty set; never clip target liberties to a cell.
fn capture_region(
    spans: &[LibertySpan],
    board: Board,
    free: &mut FreeRows<'_>,
) -> Vec<LibertySpan> {
    let Some(b) = bounds(spans) else {
        return Vec::new();
    };
    let d = board.diameter();
    if b.right - b.left >= 2 * d || b.bottom - b.top >= 2 * d {
        return Vec::new();
    }
    let witnesses: Vec<_> = DIRECTIONS
        .iter()
        .filter_map(|&v| extreme(spans, v, b.center()))
        .collect();
    if witnesses.iter().enumerate().any(|(i, &p)| {
        witnesses[..i]
            .iter()
            .any(|&q| distance(p, q) >= 4 * i64::from(d).pow(2))
    }) {
        return Vec::new();
    }
    let rect = Rect {
        left: b.right - d + 1,
        right: b.left + d - 1,
        top: b.bottom - d + 1,
        bottom: b.top + d - 1,
    };
    let sq = i64::from(d).pow(2) - 1;
    let mut covers = Vec::new();
    for y in rect.top.max(board.stone_radius())..=rect.bottom.min(1000 - board.stone_radius()) {
        let (mut left, mut right) = (rect.left, rect.right);
        for s in spans {
            let Some(w) = width(sq, y - s.y) else {
                left = 1;
                right = 0;
                break;
            };
            left = left.max(s.right - w);
            right = right.min(s.left + w);
            if left > right {
                break;
            }
        }
        if left <= right {
            covers.push(LibertySpan { y, left, right });
        }
    }
    intersect(&covers, &free.region(rect))
}
fn add(
    pool: &mut BTreeMap<u32, Candidate>,
    p: Spot,
    source: u8,
    is_frontier: bool,
    groups: &[usize],
) {
    let id = (p.x + 1000 * p.y) as u32;
    let c = pool.entry(id).or_insert_with(|| Candidate {
        action: id,
        sources: 0,
        protected: true,
        frontier: false,
        groups: Vec::new(),
        extension_groups: Vec::new(),
    });
    c.sources |= source;
    c.frontier |= is_frontier;
    c.groups.extend(groups);
    c.groups.sort_unstable();
    c.groups.dedup();
    if source & EXTEND != 0 {
        c.extension_groups.extend(groups);
        c.extension_groups.sort_unstable();
        c.extension_groups.dedup();
    }
}

pub fn generate(
    game: &Game,
    geometry: &Geometry,
    resolution: usize,
    selected: &[usize],
    budget: usize,
    seed: u64,
) -> Result<Shortlist, String> {
    generate_impl::<false>(game, geometry, resolution, selected, budget, seed)
}
pub fn generate_profiled(
    game: &Game,
    geometry: &Geometry,
    resolution: usize,
    selected: &[usize],
    budget: usize,
    seed: u64,
) -> Result<Shortlist, String> {
    generate_impl::<true>(game, geometry, resolution, selected, budget, seed)
}
/// Direct proposals: 1–3 Extend per friendly group, four oriented Join extremes
/// per group set, capture extrema, and 2–4 random slots. No padding to a budget.
/// The budget remains soft: never silently drop a selected geometric opportunity.
fn generate_impl<const PROFILE: bool>(
    game: &Game,
    geometry: &Geometry,
    resolution: usize,
    selected: &[usize],
    budget: usize,
    seed: u64,
) -> Result<Shortlist, String> {
    if budget == 0 || resolution < 2 || selected.iter().any(|&c| c >= resolution * resolution) {
        return Err("invalid geometry proposal budget/resolution/cell".into());
    }
    let timer = PROFILE.then(Instant::now);
    let mut profile = ProposalProfile::default();
    let mut selected = selected.to_vec();
    selected.sort_unstable();
    selected.dedup();
    let mut result = Shortlist {
        version: VERSION,
        selected_cells: selected.clone(),
        candidates: Vec::new(),
        preliminary_candidates: 0,
        protected_overflow: 0,
        profile: None,
    };
    let groups = &geometry.groups;
    let boxes: Vec<_> = groups.iter().map(|g| bounds(&g.spans)).collect();
    let mut free = FreeRows::new(game);
    let mut reach = ReachRows::new(game, geometry);
    let mut captures: BTreeMap<usize, Vec<LibertySpan>> = BTreeMap::new();
    let d = game.board().diameter();
    let reach_sq = 2 * i64::from(d).pow(2) - 1;
    let enemy_stones: Vec<_> = game
        .position()
        .stones()
        .iter()
        .filter(|s| s.color != game.on_move())
        .map(|s| s.spot)
        .collect();
    elapsed(timer, &mut profile.setup_seconds);
    for cell in selected {
        let timer = PROFILE.then(Instant::now);
        let rect = cell_rect(cell, resolution);
        let space = free.region(rect);
        if space.is_empty() {
            elapsed(timer, &mut profile.setup_seconds);
            continue;
        }
        let local: Vec<_> = groups.iter().map(|g| clip(&g.spans, rect)).collect();
        let friendly: Vec<_> = groups
            .iter()
            .enumerate()
            .filter(|(i, g)| g.friendly && !local[*i].is_empty())
            .map(|(i, _)| i)
            .collect();
        let enemies: Vec<_> = groups
            .iter()
            .enumerate()
            .filter(|(i, g)| !g.friendly && boxes[*i].is_some_and(|b| b.overlaps(rect.expanded(d))))
            .map(|(i, _)| i)
            .collect();
        let mut pool = BTreeMap::new();
        let mut outer_by_group = BTreeMap::new();
        elapsed(timer, &mut profile.setup_seconds);
        let timer = PROFILE.then(Instant::now);
        let contacts = union(
            enemy_stones
                .iter()
                .filter(|&&p| rect.distance(p) <= i64::from(d + 2).pow(2))
                .flat_map(|&p| contact_band(p, rect, d))
                .collect(),
        );
        for &i in &friendly {
            let sites: Vec<_> = groups[i]
                .members
                .iter()
                .map(|&k| game.position().stones()[k].spot)
                .filter(|&p| rect.distance(p) <= reach_sq)
                .collect();
            let outer = reach.outer(i, &local[i]);
            let hugging = intersect(&local[i], &contacts);
            let hug = farthest(&hugging, &sites);
            if let Some(p) = hug {
                add(&mut pool, p, EXTEND | HUG, contains(&outer, p), &[i]);
            }
            let mut extensions: Vec<_> = components(&outer)
                .iter()
                .filter_map(|piece| farthest(piece, &sites))
                .collect();
            extensions.sort_unstable_by_key(|&p| {
                (
                    std::cmp::Reverse(sites.iter().map(|&q| distance(p, q)).min().unwrap()),
                    p.x + 1000 * p.y,
                )
            });
            let mut count = usize::from(hug.is_some());
            for p in extensions {
                if Some(p) == hug {
                    continue;
                }
                if count == 3 {
                    break;
                }
                add(&mut pool, p, EXTEND, true, &[i]);
                count += 1;
            }
            outer_by_group.insert(i, outer);
        }
        elapsed(timer, &mut profile.extend_seconds);
        let timer = PROFILE.then(Instant::now);
        for (joined, region) in join_regions(&local, &friendly) {
            let origin = bounds(&region).unwrap().center();
            for direction in join_directions(game, geometry, &joined, &region) {
                if let Some(p) = extreme(&region, direction, origin) {
                    add(&mut pool, p, JOIN, false, &joined);
                }
            }
        }
        elapsed(timer, &mut profile.join_seconds);
        let timer = PROFILE.then(Instant::now);
        for &i in &enemies {
            let capture = captures
                .entry(i)
                .or_insert_with(|| capture_region(&groups[i].spans, game.board(), &mut free));
            for piece in components(&clip(capture, rect)) {
                let origin = bounds(&piece).unwrap().center();
                for direction in AXES {
                    if let Some(p) = extreme(&piece, direction, origin) {
                        add(&mut pool, p, CAPTURE, false, &[i]);
                    }
                }
            }
        }
        elapsed(timer, &mut profile.capture_seconds);
        let timer = PROFILE.then(Instant::now);
        let mut rng = StdRng::seed_from_u64(seed ^ (cell as u64).wrapping_mul(0x9e3779b97f4a7c15));
        let total: usize = space.iter().map(|s| (s.right - s.left + 1) as usize).sum();
        let slots = (budget / 4).clamp(2, 4).min(total);
        let mut sampled = std::collections::BTreeSet::new();
        for _ in 0..slots * 4 {
            if sampled.len() == slots {
                break;
            }
            let mut index = rng.random_range(0..total);
            for s in &space {
                let length = (s.right - s.left + 1) as usize;
                if index < length {
                    let p = Spot {
                        x: s.left + index as i32,
                        y: s.y,
                    };
                    sampled.insert((p.x, p.y));
                    add(&mut pool, p, COVERAGE, false, &[]);
                    break;
                }
                index -= length;
            }
        }
        elapsed(timer, &mut profile.random_seconds);
        let timer = PROFILE.then(Instant::now);
        for c in pool.values_mut() {
            let p = c.spot();
            debug_assert!(rect.contains(p));
            for &i in &friendly {
                if contains(&local[i], p) && !c.groups.contains(&i) {
                    c.groups.push(i);
                }
                if contains(&outer_by_group[&i], p) {
                    c.frontier = true;
                }
            }
            c.groups.sort_unstable();
            if PROFILE {
                for (n, bit) in profile
                    .final_source_counts
                    .iter_mut()
                    .zip([EXTEND, JOIN, HUG, CAPTURE, COVERAGE])
                {
                    *n += u64::from(c.sources & bit != 0);
                }
            }
        }
        result.protected_overflow += pool.len().saturating_sub(budget);
        result.preliminary_candidates += pool.len();
        result.candidates.extend(pool.into_values());
        elapsed(timer, &mut profile.tag_seconds);
    }
    let timer = PROFILE.then(Instant::now);
    result.candidates.sort_unstable_by_key(|c| c.action);
    debug_assert!(result.candidates.iter().all(|c| {
        result
            .selected_cells
            .binary_search(&cells::cell(c.action, resolution))
            .is_ok()
    }));
    elapsed(timer, &mut profile.finish_seconds);
    if PROFILE {
        result.profile = Some(profile)
    }
    Ok(result)
}

#[cfg(test)]
mod tests;
