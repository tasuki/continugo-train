//! SGF uses SZ, PB/PW and scored RE for standard game information.
//! Root JSON contains seed, search/model settings (when used) and optional profiling.
//! Move comments carry pre-move policy targets and/or profiling; otherwise they
//! are omitted. Truncated games have no RE. Generation/export timings finish
//! after annotation serialization, so those remain stderr-only.

pub mod parallel;

use crate::{
    action::random_move,
    inference::Evaluator,
    profile::{GameProfile, MoveProfile},
    search::{SearchConfig, Visit, search, search_neural},
};
use continugo_analysis::GeometricAvailability;
use continugo_rules::{Board, Color, Game, sgf, territory};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::Serialize;
use std::time::Instant;

#[derive(Clone, Copy, Debug, Serialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Engine {
    Random,
    Mcts,
    Neural,
}

#[derive(Clone, Debug, Serialize)]
pub struct SelfplayConfig {
    pub engine: Engine,
    pub max_moves: usize,
    pub search: SearchConfig,
}

impl SelfplayConfig {
    pub fn validate(&self) -> Result<(), String> {
        self.search.validate()?;
        if self.max_moves == 0 {
            return Err("max_moves must be positive".into());
        }
        Ok(())
    }
}

/// In-memory generation data, not a duplicate JSON representation of the SGF.
#[derive(Debug)]
pub struct GameMetadata {
    pub seed: u64,
    pub board_size: i32,
    pub config: SelfplayConfig,
    /// One entry per SGF move: sparse pre-move root visits, or None for random play.
    pub policies: Vec<Option<Vec<Visit>>>,
    /// None for a truncated game; otherwise the authoritative scored outcome.
    pub result: Option<territory::Outcome>,
    pub profile: Option<GameProfile>,
    pub model: Option<String>,
    pub opponent: Option<String>,
}

#[derive(Serialize)]
struct GameComment<'a> {
    seed: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    search: Option<&'a SearchConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    profile: Option<&'a GameProfile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    opponent: Option<&'a str>,
}

impl GameMetadata {
    fn comment(&self) -> GameComment<'_> {
        GameComment {
            seed: self.seed,
            search: (!matches!(self.config.engine, Engine::Random)).then_some(&self.config.search),
            model: self.model.as_deref(),
            profile: self.profile.as_ref(),
            opponent: self.opponent.as_deref(),
        }
    }
}

#[derive(Serialize)]
struct MoveComment<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    policy: Option<&'a [Visit]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    profile: Option<&'a MoveProfile>,
}

#[derive(Debug)]
pub struct GameRecord {
    pub sgf: String,
    pub metadata: GameMetadata,
}

pub fn generate(board: Board, config: &SelfplayConfig, seed: u64) -> Result<GameRecord, String> {
    generate_inner::<false>(board, config, seed, None, None)
}

pub fn generate_profiled(
    board: Board,
    config: &SelfplayConfig,
    seed: u64,
) -> Result<GameRecord, String> {
    if !matches!(config.engine, Engine::Random) {
        return Err("Detailed profiling currently supports --engine random only".into());
    }
    generate_inner::<true>(board, config, seed, None, None)
}

pub fn generate_neural(
    board: Board,
    config: &SelfplayConfig,
    seed: u64,
    evaluator: &mut dyn Evaluator,
) -> Result<GameRecord, String> {
    if !matches!(config.engine, Engine::Neural) {
        return Err("neural evaluator requires neural engine".into());
    }
    generate_inner::<false>(board, config, seed, Some(evaluator), None)
}

/// Color-specific frozen networks, same search budget/rules. Evaluation chooses
/// maximum visits (random tie breaks), not training's temperature-one sampling.
pub fn generate_match(
    board: Board,
    config: &SelfplayConfig,
    seed: u64,
    black: &mut dyn Evaluator,
    white: &mut dyn Evaluator,
) -> Result<GameRecord, String> {
    if !matches!(config.engine, Engine::Neural) {
        return Err("matches require the neural engine".into());
    }
    generate_inner::<false>(board, config, seed, Some(black), Some(white))
}

fn generate_inner<const PROFILE: bool>(
    board: Board,
    config: &SelfplayConfig,
    seed: u64,
    mut evaluator: Option<&mut dyn Evaluator>,
    mut opponent: Option<&mut dyn Evaluator>,
) -> Result<GameRecord, String> {
    config.validate()?;
    if matches!(config.engine, Engine::Neural) && evaluator.is_none() {
        return Err("neural engine requires a loaded model".into());
    }
    let model = evaluator.as_ref().map(|e| e.model_id().to_owned());
    let opponent_id = opponent.as_ref().map(|e| e.model_id().to_owned());
    let name = evaluator.as_ref().map(|e| e.network_name().to_owned());
    let opponent_name = opponent.as_ref().map(|e| e.network_name().to_owned());
    let evaluation_game = opponent.is_some();
    let started = PROFILE.then(Instant::now);
    let mut profile = GameProfile::default();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut game = Game::new(board);
    let index_started = PROFILE.then(Instant::now);
    let mut availability = matches!(config.engine, Engine::Random)
        .then(|| GeometricAvailability::new(game.position()));
    if let Some(started) = index_started {
        profile.availability_init_seconds = started.elapsed().as_secs_f64();
    }
    let mut policies = Vec::new();
    while !game.is_finished() && policies.len() < config.max_moves {
        let move_started = PROFILE.then(Instant::now);
        let mut move_profile = MoveProfile {
            ply: policies.len() + 1,
            stones_before: game.position().stones().len(),
            geometric_candidates: availability
                .as_ref()
                .map_or(0, |index| index.coordinates().len()),
            ..MoveProfile::default()
        };
        let (next, policy) = match config.engine {
            Engine::Random => {
                let candidate = random_move::<PROFILE>(
                    &game,
                    availability.as_ref().expect("random engine index"),
                    &mut rng,
                    &mut move_profile,
                );
                let next = match candidate {
                    Some((_, next)) => next,
                    None => game.pass()?,
                };
                (next, None)
            }
            Engine::Mcts => {
                let result = search(&game, &config.search, &mut rng)?;
                (result.sample(&mut rng).apply(&game)?, Some(result.policy))
            }
            Engine::Neural => {
                let result = search_neural(
                    &game,
                    &config.search,
                    if evaluation_game && game.on_move() == Color::White {
                        opponent.as_deref_mut().expect("opposing evaluator")
                    } else {
                        evaluator.as_deref_mut().expect("validated evaluator")
                    },
                    &mut rng,
                )?;
                let action = if evaluation_game {
                    result
                        .policy
                        .iter()
                        .max_by_key(|visit| (visit.visits, rng.random::<u64>()))
                        .unwrap()
                        .action
                } else {
                    result.sample(&mut rng)
                };
                (action.apply(&game)?, Some(result.policy))
            }
        };
        if let Some(index) = &mut availability {
            let update_started = PROFILE.then(Instant::now);
            index.sync(next.position());
            if let Some(started) = update_started {
                move_profile.availability_update.record(started);
            }
        }
        game = next;
        policies.push(policy);
        if let Some(move_started) = move_started {
            move_profile.seconds = move_started.elapsed().as_secs_f64();
            move_profile.passed =
                matches!(game.record().last(), Some(continugo_rules::Play::Pass(_)));
            if move_profile.seconds >= 1.0 {
                eprintln!(
                    "profile move {}: {:.3}s, {} proposals, rules={:.3}s, eyes={:.3}s, pass={}",
                    move_profile.ply,
                    move_profile.seconds,
                    move_profile.proposals,
                    move_profile.rules.seconds,
                    move_profile.eyes.seconds,
                    move_profile.passed,
                );
            }
            profile.moves.push(move_profile);
        }
    }
    if let Some(started) = started {
        profile.selfplay_seconds = started.elapsed().as_secs_f64();
    }
    let finished = game.is_finished();
    let scoring_started = PROFILE.then(Instant::now);
    let result = finished.then(|| territory::result(territory::areas(game.position().stones())));
    if let Some(started) = scoring_started {
        profile.scoring_seconds = started.elapsed().as_secs_f64();
    }
    let sgf_started = PROFILE.then(Instant::now);
    let mut metadata = GameMetadata {
        seed,
        board_size: board.size(),
        config: config.clone(),
        policies,
        result,
        profile: PROFILE.then_some(profile),
        model,
        opponent: opponent_id,
    };
    let mut record = sgf::Record {
        plays: game.record().to_vec(),
        ..sgf::Record::default()
    };
    record
        .root
        .iter_mut()
        .find(|(name, _)| name == "SZ")
        .expect("default SGF root includes board size")
        .1 = vec![board.size().to_string()];
    let player = match config.engine {
        Engine::Random => "random".to_owned(),
        Engine::Mcts => "mcts".to_owned(),
        Engine::Neural => format!("neural-{}", name.as_deref().unwrap()),
    };
    record.root.extend([
        ("PB".into(), vec![player.clone()]),
        (
            "PW".into(),
            vec![
                opponent_name
                    .as_ref()
                    .map_or(player, |id| format!("neural-{id}")),
            ],
        ),
    ]);
    if let Some(result) = result {
        record
            .root
            .push(("RE".into(), vec![sgf::format_result(board, result)]));
    }
    record.root.push((
        "C".into(),
        vec![serde_json::to_string(&metadata.comment()).map_err(|e| e.to_string())?],
    ));
    for (index, policy) in metadata.policies.iter().enumerate() {
        let comment = MoveComment {
            policy: policy.as_deref(),
            profile: metadata.profile.as_ref().map(|p| &p.moves[index]),
        };
        let properties = if comment.policy.is_none() && comment.profile.is_none() {
            Vec::new()
        } else {
            vec![(
                "C".into(),
                vec![serde_json::to_string(&comment).map_err(|e| e.to_string())?],
            )]
        };
        record.move_properties.push(properties);
    }
    let sgf = sgf::encode_record(&record)?;
    if let Some(profile) = &mut metadata.profile {
        profile.sgf_seconds = sgf_started
            .expect("profile export timer")
            .elapsed()
            .as_secs_f64();
        profile.seconds = started.expect("profile game timer").elapsed().as_secs_f64();
    }
    Ok(GameRecord { sgf, metadata })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_use_both_color_specific_evaluators_and_authoritative_draws() {
        struct PassOnly {
            id: &'static str,
            calls: usize,
        }
        impl Evaluator for PassOnly {
            fn model_id(&self) -> &str {
                self.id
            }
            fn evaluate(&mut self, _: &Game) -> Result<crate::inference::Evaluation, String> {
                self.calls += 1;
                let mut logits = vec![f32::NEG_INFINITY; 1_000_001];
                logits[1_000_000] = 0.0;
                Ok(crate::inference::Evaluation {
                    value: 0.0,
                    logits,
                    legal: vec![0; 1_000_000],
                })
            }
        }
        let mut black = PassOnly {
            id: "black",
            calls: 0,
        };
        let mut white = PassOnly {
            id: "white",
            calls: 0,
        };
        let config = SelfplayConfig {
            engine: Engine::Neural,
            max_moves: 8,
            search: SearchConfig {
                simulations: 4,
                ..SearchConfig::default()
            },
        };
        let game =
            generate_match(Board::new(9).unwrap(), &config, 1, &mut black, &mut white).unwrap();
        assert!(black.calls > 0 && white.calls > 0);
        assert!(matches!(
            game.metadata.result,
            Some(territory::Outcome::Draw)
        ));
        assert!(game.sgf.contains("PB[neural-black]"));
        assert!(game.sgf.contains("PW[neural-white]"));
        assert!(game.sgf.contains("RE[0]"));
        let info = crate::replay::ReplayGame::load(&game.sgf)
            .unwrap()
            .info(4)
            .unwrap();
        assert_eq!(info.result.as_deref(), Some("0"));
        assert_eq!(info.policy_visit_totals, vec![Some(4), Some(4)]);
    }
    use continugo_analysis::fills_small_eye;
    use continugo_rules::Play;

    fn config(engine: Engine) -> SelfplayConfig {
        SelfplayConfig {
            engine,
            max_moves: 8,
            search: SearchConfig {
                simulations: 8,
                ..SearchConfig::default()
            },
        }
    }

    fn replay(record: &GameRecord) -> Game {
        let decoded = sgf::decode_record(&record.sgf).unwrap();
        let size = decoded
            .root
            .iter()
            .find(|(name, _)| name == "SZ")
            .unwrap()
            .1[0]
            .parse::<i32>()
            .unwrap();
        let plays = decoded.plays;
        let metadata = &record.metadata;
        assert_eq!(
            decoded.root.iter().find(|(name, _)| name == "C").unwrap().1[0],
            serde_json::to_string(&metadata.comment()).unwrap()
        );
        assert_eq!(decoded.move_properties.len(), plays.len());
        assert_eq!(size, metadata.board_size);
        let board = Board::new(size).unwrap();
        let mut game = Game::new(board);
        assert_eq!(plays.len(), metadata.policies.len());
        for (index, (play, policy)) in plays.iter().zip(&metadata.policies).enumerate() {
            let comment = MoveComment {
                policy: policy.as_deref(),
                profile: metadata.profile.as_ref().map(|p| &p.moves[index]),
            };
            if policy.is_none() && metadata.profile.is_none() {
                assert!(decoded.move_properties[index].is_empty());
            } else {
                assert_eq!(
                    decoded.move_properties[index],
                    vec![("C".into(), vec![serde_json::to_string(&comment).unwrap()])]
                );
            }
            let next = match play {
                Play::Place(stone) => game.play(stone.spot),
                Play::Pass(_) => game.pass(),
            }
            .unwrap();
            if matches!(metadata.config.engine, Engine::Random)
                && let Play::Place(stone) = play
            {
                let enemy = stone.color.other();
                let enemies_before = game
                    .position()
                    .stones()
                    .iter()
                    .filter(|s| s.color == enemy)
                    .count();
                let enemies_after = next
                    .position()
                    .stones()
                    .iter()
                    .filter(|s| s.color == enemy)
                    .count();
                assert!(
                    enemies_after < enemies_before
                        || !fills_small_eye(game.position(), stone.color, stone.spot)
                );
            }
            if let Some(policy) = policy {
                assert_eq!(
                    policy.iter().map(|entry| entry.visits).sum::<u32>(),
                    metadata.config.search.simulations
                );
                for entry in policy {
                    assert!(entry.action.apply(&game).is_ok());
                }
                assert!(policy.iter().any(|entry| entry.visits > 0
                    && entry.action.apply(&game).unwrap() == next));
            }
            game = next;
        }
        assert_eq!(Game::from_record_on(board, &plays).unwrap(), game);
        game
    }

    #[test]
    fn random_and_mcts_are_replayable_and_seeded_across_sizes() {
        for engine in [Engine::Random, Engine::Mcts] {
            for size in [9, 13, 19] {
                let board = Board::new(size).unwrap();
                let a = generate(board, &config(engine), 123).unwrap();
                let b = generate(board, &config(engine), 123).unwrap();
                assert_eq!(a.sgf, b.sgf);

                let game = replay(&a);
                if matches!(engine, Engine::Random) {
                    assert!(
                        game.record()
                            .iter()
                            .all(|play| matches!(play, Play::Place(_)))
                    );
                }
                assert_eq!(game.record().len(), a.metadata.policies.len());
                assert_eq!(game.is_finished(), a.metadata.result.is_some());
            }
        }
    }

    #[test]
    fn profiling_preserves_moves_and_accounts_for_work() {
        let config = config(Engine::Random);
        let plain = generate(Board::default(), &config, 123).unwrap();
        let timed = generate_profiled(Board::default(), &config, 123).unwrap();
        assert_eq!(
            sgf::decode(&plain.sgf).unwrap(),
            sgf::decode(&timed.sgf).unwrap()
        );
        replay(&plain);
        replay(&timed);
        let profile = timed.metadata.profile.as_ref().unwrap();
        assert_eq!(profile.moves.len(), timed.metadata.policies.len());
        assert!(profile.seconds >= profile.selfplay_seconds);
        for m in &profile.moves {
            assert_eq!(m.rules_preparation.calls, 1);
            assert_eq!(m.rules.calls, m.proposals);
            assert_eq!(m.capture_check.calls, m.legal);
            assert!(m.rejected_eyes <= m.eyes.calls);
            assert_eq!(m.legal, m.rejected_eyes + u64::from(!m.passed));
            assert!(m.proposals <= m.geometric_candidates as u64);
            assert_eq!(m.availability_update.calls, 1);
            if m.passed {
                assert_eq!(m.proposals, m.geometric_candidates as u64);
            }
            assert!(
                m.seconds
                    >= m.rules_preparation.seconds
                        + m.rules.seconds
                        + m.capture_check.seconds
                        + m.eyes.seconds
                        + m.availability_update.seconds
            );
        }
        let mut metadata = serde_json::to_value(timed.metadata.comment()).unwrap();
        metadata.as_object_mut().unwrap().remove("profile");
        assert_eq!(
            metadata,
            serde_json::to_value(plain.metadata.comment()).unwrap()
        );
    }

    #[test]
    fn only_natural_completion_gets_an_outcome_label() {
        let mut config = config(Engine::Random);
        // Size 1 has no legal placements, so a genuine exhaustive scan must pass.
        let board = Board::new(1).unwrap();
        let finished = generate(board, &config, 0).unwrap();
        assert_eq!(
            sgf::decode(&finished.sgf).unwrap(),
            sgf::decode("B[];W[]").unwrap()
        );
        assert_eq!(finished.metadata.result, Some(territory::Outcome::Draw));
        let finished_sgf = sgf::decode_record(&finished.sgf).unwrap();
        assert_eq!(
            finished_sgf
                .root
                .iter()
                .find(|(name, _)| name == "RE")
                .unwrap()
                .1,
            vec!["0"]
        );
        config.max_moves = 1;
        let truncated = generate(board, &config, 0).unwrap();
        assert_eq!(
            sgf::decode(&truncated.sgf).unwrap(),
            sgf::decode("B[]").unwrap()
        );
        assert_eq!(truncated.metadata.result, None);
        assert!(
            !sgf::decode_record(&truncated.sgf)
                .unwrap()
                .root
                .iter()
                .any(|(name, _)| name == "RE")
        );
        assert_eq!(
            serde_json::to_value(truncated.metadata.comment()).unwrap(),
            serde_json::json!({"seed": 0})
        );
    }
}
