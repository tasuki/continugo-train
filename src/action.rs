use crate::profile::MoveProfile;
use continugo_analysis::{GeometricAvailability, SmallEyeAnalysis};
pub use continugo_rules::{Action, COORDINATE_ACTIONS};
use continugo_rules::{Game, Play, PreparedGame};
use rand::Rng;
use std::time::Instant;

#[cfg(test)]
mod analysis_bench;

/// An affine permutation without an allocation. Empty/singleton domains are valid.
/// Every index is visited once; this is NOT a uniformly random permutation.
struct Permutation {
    next: u32,
    step: u32,
    remaining: u32,
    modulus: u32,
}

impl Permutation {
    fn new(length: u32, rng: &mut impl Rng) -> Self {
        if length <= 1 {
            return Self {
                next: 0,
                step: 1,
                remaining: length,
                modulus: length,
            };
        }
        let step = loop {
            let step = rng.random_range(1..length);
            if gcd(step, length) == 1 {
                break step;
            }
        };
        Self {
            next: rng.random_range(0..length),
            step,
            remaining: length,
            modulus: length,
        }
    }
}

impl Iterator for Permutation {
    type Item = u32;

    fn next(&mut self) -> Option<u32> {
        if self.remaining == 0 {
            return None;
        }
        let index = self.next;
        self.next = (self.next + self.step) % self.modulus;
        self.remaining -= 1;
        Some(index)
    }
}

fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// MCTS proposals still cover the full coordinate domain with bounded work.
pub(crate) struct Proposals {
    order: Permutation,
}

impl Proposals {
    pub fn new(rng: &mut impl Rng) -> Self {
        Self {
            order: Permutation::new(COORDINATE_ACTIONS, rng),
        }
    }

    pub fn next(&mut self) -> Option<Action> {
        self.order
            .next()
            .map(|index| Action::from_index(index).expect("coordinate index"))
    }

    /// Failure means the bounded MCTS budget was spent, NOT no legal move.
    pub fn legal(&mut self, game: &PreparedGame, attempts: usize) -> Option<(Action, Game)> {
        self.legal_accepting(game, attempts, |_, _| true)
    }

    /// Bootstrap search guidance only: legality is unchanged, captures are exempt.
    pub fn bootstrap_legal(
        &mut self,
        game: &Game,
        prepared: &PreparedGame,
        attempts: usize,
    ) -> Option<(Action, Game)> {
        let mut eyes = SmallEyeAnalysis::new(game.position());
        self.legal_accepting(prepared, attempts, |action, next| {
            next.position().stones().len() < game.position().stones().len() + 1
                || !eyes.fills_small_eye(game.on_move(), action.spot().unwrap())
        })
    }

    fn legal_accepting(
        &mut self,
        game: &PreparedGame,
        attempts: usize,
        mut accept: impl FnMut(Action, &Game) -> bool,
    ) -> Option<(Action, Game)> {
        for _ in 0..attempts {
            let action = self.next()?;
            if let Ok(next) = game.play(action.spot().expect("coordinate proposal"))
                && accept(action, &next)
            {
                return Some((action, next));
            }
        }
        None
    }
}

/// Exhaust the current geometric free list before reporting no acceptable move.
/// Pocket geometry is reused within this position; color and candidate-specific
/// filling/connection checks remain independent. Nothing survives a placement.
pub(crate) fn random_move<const PROFILE: bool>(
    game: &Game,
    availability: &GeometricAvailability,
    rng: &mut impl Rng,
    profile: &mut MoveProfile,
) -> Option<(Action, Game)> {
    let coordinates = availability.coordinates();
    let order = Permutation::new(coordinates.len() as u32, rng);
    let mut eyes = SmallEyeAnalysis::with_availability(game.position(), availability);
    select_random_move::<PROFILE>(
        game,
        order.map(|index| coordinates[index as usize]),
        profile,
        &mut eyes,
    )
}

fn select_random_move<const PROFILE: bool>(
    game: &Game,
    candidates: impl Iterator<Item = Action>,
    profile: &mut MoveProfile,
    eyes: &mut SmallEyeAnalysis<'_>,
) -> Option<(Action, Game)> {
    let started = PROFILE.then(Instant::now);
    let prepared = game.prepare();
    if let Some(started) = started {
        profile.rules_preparation.record(started);
    }
    let enemy = game.on_move().other();
    let enemy_count = game
        .position()
        .stones()
        .iter()
        .filter(|s| s.color == enemy)
        .count();
    for action in candidates {
        let started = PROFILE.then(Instant::now);
        let result = prepared.play(action.spot().expect("coordinate proposal"));
        if let Some(started) = started {
            profile.proposals += 1;
            let seconds = profile.rules.record(started);
            if result.is_ok() {
                profile.legal_rules_seconds += seconds;
            } else {
                profile.illegal_rules_seconds += seconds;
            }
        }
        if let Ok(next) = result {
            if PROFILE {
                profile.legal += 1;
            }
            let Play::Place(stone) = *next.record().last().expect("placement recorded") else {
                unreachable!("coordinate proposals exclude pass");
            };
            let started = PROFILE.then(Instant::now);
            let captured = next
                .position()
                .stones()
                .iter()
                .filter(|s| s.color == enemy)
                .count()
                < enemy_count;
            if let Some(started) = started {
                profile.capture_check.record(started);
            }
            if captured {
                return Some((action, next));
            }
            let started = PROFILE.then(Instant::now);
            let eye = eyes.fills_small_eye(stone.color, stone.spot);
            if let Some(started) = started {
                profile.eyes.record(started);
            }
            if !eye {
                return Some((action, next));
            }
            if PROFILE {
                profile.rejected_eyes += 1;
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use continugo_rules::{Board, Color, Spot, Stone, sgf};
    use rand::{SeedableRng, rngs::StdRng};

    fn try_one(game: &Game, index: u32) -> Option<(Action, Game)> {
        select_random_move::<false>(
            game,
            std::iter::once(Action::from_index(index).unwrap()),
            &mut MoveProfile::default(),
            &mut SmallEyeAnalysis::new(game.position()),
        )
    }

    #[test]
    fn random_search_continues_past_the_old_budget() {
        let game = Game::new(Board::new(2).unwrap());
        let available = GeometricAvailability::new(game.position());
        // Start at (500,500): many geometrically free centres are still suicide.
        let first = available
            .coordinates()
            .iter()
            .position(|a| a.index() == 500_500)
            .unwrap();
        let order = Permutation {
            next: first as u32,
            step: 1,
            remaining: available.coordinates().len() as u32,
            modulus: available.coordinates().len() as u32,
        };
        let mut profile = MoveProfile::default();
        let result = select_random_move::<true>(
            &game,
            order.map(|i| available.coordinates()[i as usize]),
            &mut profile,
            &mut SmallEyeAnalysis::with_availability(game.position(), &available),
        );
        assert!(result.is_some());
        assert!(profile.proposals > 128);
    }

    #[test]
    fn random_search_checks_only_geometrically_free_coordinates_before_passing() {
        let game = Game::new(Board::new(1).unwrap());
        let available = GeometricAvailability::new(game.position());
        let mut profile = MoveProfile::default();
        assert!(
            random_move::<true>(
                &game,
                &available,
                &mut StdRng::seed_from_u64(0),
                &mut profile
            )
            .is_none()
        );
        assert_eq!(available.coordinates().len(), 1);
        assert_eq!(profile.proposals, 1); // Not a million boundary rejections.
    }

    #[test]
    fn random_search_does_not_fill_a_friendly_eye() {
        let mut game = Game::new(Board::new(3).unwrap());
        for (x, y) in [
            (498, 166),
            (830, 166),
            (166, 498),
            (498, 498),
            (830, 498),
            (166, 830),
            (498, 830),
        ] {
            game = game.play(Spot { x, y }).unwrap().pass().unwrap();
        }
        let spot = Spot { x: 166, y: 166 };
        assert!(game.play(spot).is_ok());
        assert!(
            game.position()
                .play(Stone {
                    color: Color::White,
                    spot
                })
                .is_none()
        );
        assert!(try_one(&game, 166_166).is_none());
        let available = GeometricAvailability::new(game.position());
        let mut profile = MoveProfile::default();
        assert!(
            random_move::<true>(
                &game,
                &available,
                &mut StdRng::seed_from_u64(0),
                &mut profile
            )
            .is_none()
        );
        assert_eq!(profile.proposals, available.coordinates().len() as u64);
    }

    #[test]
    fn larger_pockets_and_connecting_moves_remain_available() {
        for stones in [
            vec![
                (830, 166),
                (166, 498),
                (498, 498),
                (830, 498),
                (166, 830),
                (498, 830),
            ],
            vec![(166, 498), (498, 166), (830, 498), (498, 830)],
        ] {
            let mut game = Game::new(Board::new(3).unwrap());
            for (x, y) in stones {
                game = game.play(Spot { x, y }).unwrap().pass().unwrap();
            }
            let index = if game.position().stones().len() == 6 {
                166_166
            } else {
                498_498
            };
            assert!(try_one(&game, index).is_some());
        }
    }

    #[test]
    fn capturing_moves_remain_available() {
        let fixture = include_str!("../../continugo/fixtures/rules/capture-four.txt");
        let record = fixture
            .lines()
            .nth(1)
            .unwrap()
            .strip_prefix("record: ")
            .unwrap();
        let game = Game::from_record(&sgf::decode(record).unwrap()).unwrap();
        let (_, next) = try_one(&game, 240_839).unwrap(); // B[qheG]
        assert_eq!(
            next.position().stones().len(),
            game.position().stones().len() - 3
        );
    }

    fn bootstrap_one(game: &Game, index: u32) -> Option<(Action, Game)> {
        let mut proposals = Proposals {
            order: Permutation {
                next: index,
                step: 1,
                remaining: 1,
                modulus: COORDINATE_ACTIONS,
            },
        };
        proposals.bootstrap_legal(game, &game.prepare(), 1)
    }

    #[test]
    fn bootstrap_guidance_rejects_eyes_but_keeps_connections_and_captures() {
        let mut eye = Game::new(Board::new(3).unwrap());
        for (x, y) in [
            (498, 166),
            (830, 166),
            (166, 498),
            (498, 498),
            (830, 498),
            (166, 830),
            (498, 830),
        ] {
            eye = eye.play(Spot { x, y }).unwrap().pass().unwrap();
        }
        assert!(eye.play(Spot { x: 166, y: 166 }).is_ok());
        assert!(bootstrap_one(&eye, 166_166).is_none());
        // The opponent can fill that same pocket to capture after its protection
        // is removed. Tactical eye fills must not be mistaken for own-eye waste.
        let attack = eye.play(Spot { x: 830, y: 830 }).unwrap();
        let (_, next) = bootstrap_one(&attack, 166_166).unwrap();
        assert_eq!(next.position().stones().len(), 1);

        let mut connection = Game::new(Board::new(3).unwrap());
        for (x, y) in [(166, 498), (498, 166), (830, 498), (498, 830)] {
            connection = connection.play(Spot { x, y }).unwrap().pass().unwrap();
        }
        assert!(bootstrap_one(&connection, 498_498).is_some());
        let fixture = include_str!("../../continugo/fixtures/rules/capture-four.txt");
        let record = fixture
            .lines()
            .find_map(|l| l.strip_prefix("record: "))
            .unwrap();
        let capture = Game::from_record(&sgf::decode(record).unwrap()).unwrap();
        assert!(bootstrap_one(&capture, 240_839).is_some());
    }

    fn benchmark_game() -> Game {
        let path = std::env::var("CONTINUGO_BENCH_SGF").expect("set CONTINUGO_BENCH_SGF");
        let record = sgf::decode_record(&std::fs::read_to_string(path).unwrap()).unwrap();
        let size = record.root.iter().find(|(name, _)| name == "SZ").unwrap().1[0]
            .parse()
            .unwrap();
        let ply = std::env::var("CONTINUGO_BENCH_PLY")
            .ok()
            .map(|n| n.parse::<usize>().unwrap() - 1)
            .unwrap_or_else(|| {
                record
                    .plays
                    .iter()
                    .position(|p| matches!(p, Play::Pass(_)))
                    .unwrap()
            });
        Game::from_record_on(Board::new(size).unwrap(), &record.plays[..ply]).unwrap()
    }

    #[test]
    #[ignore = "explicit release benchmark; set CONTINUGO_BENCH_SGF"]
    fn benchmark_rule_internals() {
        let game = benchmark_game();
        let size = game.board().size();
        let ply = game.record().len();
        let available = GeometricAvailability::new(game.position());
        let mut profile = continugo_rules::position::PlayProfile::default();
        #[cfg(feature = "reference-rules")]
        let mut reference = continugo_rules::position::PlayProfile::default();
        let mut legal = 0;
        for action in available.coordinates() {
            let id = action.index();
            let stone = Stone {
                color: game.on_move(),
                spot: Spot {
                    x: (id % 1000) as i32,
                    y: (id / 1000) as i32,
                },
            };
            let next = game.position().play_profiled(stone, &mut profile);
            legal += usize::from(next.is_some());
            #[cfg(feature = "reference-rules")]
            assert_eq!(
                next,
                game.position()
                    .play_reference_profiled(stone, &mut reference),
                "candidate {id}"
            );
        }
        println!(
            "size={size}, ply={}, stones={}, legal={legal}",
            ply + 1,
            game.position().stones().len()
        );
        println!("optimized:\n{}", profile.summary());
        #[cfg(feature = "reference-rules")]
        println!("exhaustive reference:\n{}", reference.summary());
    }

    #[test]
    #[ignore = "explicit release benchmark; set CONTINUGO_BENCH_SGF"]
    fn benchmark_prepared_rules() {
        let game = benchmark_game();
        let available = GeometricAvailability::new(game.position());
        let started = Instant::now();
        let prepared = game.position().prepare();
        let preparation = started.elapsed().as_secs_f64();
        let mut queries = 0;
        let mut ordinary_seconds = 0.0;
        let mut prepared_seconds = 0.0;
        for color in [Color::Black, Color::White] {
            let stones: Vec<_> = available
                .coordinates()
                .iter()
                .map(|action| Stone {
                    color,
                    spot: action.spot().unwrap(),
                })
                .collect();
            let started = Instant::now();
            let ordinary: Vec<_> = stones
                .iter()
                .map(|&stone| game.position().play(stone))
                .collect();
            ordinary_seconds += started.elapsed().as_secs_f64();
            let started = Instant::now();
            let cached: Vec<_> = stones.iter().map(|&stone| prepared.play(stone)).collect();
            prepared_seconds += started.elapsed().as_secs_f64();
            for ((stone, ordinary), cached) in stones.iter().zip(ordinary).zip(cached) {
                assert_eq!(ordinary, cached, "{stone:?}");
                queries += 1;
            }
        }
        println!(
            "size={}, ply={}, stones={}, queries={queries}, preparation={preparation:.6}s, ordinary={ordinary_seconds:.6}s, prepared={prepared_seconds:.6}s",
            game.board().size(),
            game.record().len() + 1,
            game.position().stones().len()
        );
    }

    #[test]
    #[ignore = "explicit release benchmark; set CONTINUGO_BENCH_SGF"]
    fn benchmark_eye_cache() {
        let game = benchmark_game();
        let available = GeometricAvailability::new(game.position());
        let mut results = Vec::new();
        let started = Instant::now();
        let mut eyes = SmallEyeAnalysis::with_availability(game.position(), &available);
        for color in [game.on_move(), game.on_move().other()] {
            for action in available.coordinates() {
                let id = action.index() as i32;
                let spot = Spot {
                    x: id % 1000,
                    y: id / 1000,
                };
                results.push((color, spot, eyes.fills_small_eye(color, spot)));
            }
        }
        let seconds = started.elapsed().as_secs_f64();
        println!(
            "size={}, ply={}, stones={}, {} queries, {} eye fillings, cached={seconds:.6}s",
            game.board().size(),
            game.record().len() + 1,
            game.position().stones().len(),
            results.len(),
            results.iter().filter(|(_, _, eye)| *eye).count()
        );
        #[cfg(feature = "reference-eyes")]
        {
            let started = Instant::now();
            for (color, spot, expected) in results {
                assert_eq!(
                    expected,
                    continugo_analysis::eyes::fills_small_eye_reference(
                        game.position(),
                        color,
                        spot
                    ),
                    "{color:?}, {spot:?}"
                );
            }
            println!(
                "original per-candidate flood fill={:.6}s",
                started.elapsed().as_secs_f64()
            );
        }
    }

    /// Fixed-position comparison: candidate order cannot affect an exhausted scan.
    #[test]
    #[ignore = "explicit release benchmark; set CONTINUGO_BENCH_SGF to a completed random game"]
    fn benchmark_cached_endgame_scan() {
        let path = std::env::var("CONTINUGO_BENCH_SGF")
            .expect("set CONTINUGO_BENCH_SGF to a completed random-game SGF");
        let record = sgf::decode_record(&std::fs::read_to_string(path).unwrap()).unwrap();
        let size = record.root.iter().find(|(name, _)| name == "SZ").unwrap().1[0]
            .parse()
            .unwrap();
        let first_pass = record
            .plays
            .iter()
            .position(|play| matches!(play, Play::Pass(_)))
            .unwrap();
        let game =
            Game::from_record_on(Board::new(size).unwrap(), &record.plays[..first_pass]).unwrap();
        let started = Instant::now();
        let available = GeometricAvailability::new(game.position());
        let initialization = started.elapsed().as_secs_f64();
        let mut rng = StdRng::seed_from_u64(0);
        let mut all = Proposals::new(&mut rng);
        let mut old = MoveProfile::default();
        let started = Instant::now();
        assert!(
            select_random_move::<true>(
                &game,
                std::iter::from_fn(|| all.next()),
                &mut old,
                &mut SmallEyeAnalysis::new(game.position()),
            )
            .is_none()
        );
        let old_seconds = started.elapsed().as_secs_f64();
        let mut cached = MoveProfile::default();
        let started = Instant::now();
        assert!(random_move::<true>(&game, &available, &mut rng, &mut cached).is_none());
        let cached_seconds = started.elapsed().as_secs_f64();
        assert_eq!(cached.proposals, available.coordinates().len() as u64);
        assert_eq!(old.legal, cached.legal);
        assert_eq!(old.rejected_eyes, cached.rejected_eyes);
        println!(
            "size={size}, ply={}, stones={}, index build={initialization:.6}s",
            first_pass + 1,
            game.position().stones().len()
        );
        println!(
            "old: {old_seconds:.6}s, {} proposals, rules={:.6}s, eyes={:.6}s",
            old.proposals, old.rules.seconds, old.eyes.seconds
        );
        println!(
            "indexed: {cached_seconds:.6}s, {} proposals, rules={:.6}s, eyes={:.6}s",
            cached.proposals, cached.rules.seconds, cached.eyes.seconds
        );
    }

    #[test]
    fn permutations_cover_each_index_once_including_empty_and_singleton_domains() {
        let mut rng = StdRng::seed_from_u64(42);
        for length in [0, 1, 2, 6, 7, 100, COORDINATE_ACTIONS] {
            let mut order = Permutation::new(length, &mut rng);
            let mut seen = vec![false; length as usize];
            for _ in 0..length {
                let index = order.next().unwrap() as usize;
                assert!(!seen[index]);
                seen[index] = true;
            }
            assert!(order.next().is_none());
            assert!(seen.into_iter().all(|present| present));
        }
    }
}
