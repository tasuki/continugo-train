//! SGF uses SZ, PB/PW and scored RE for standard game information.
//! Root JSON contains seed, search/model settings (when used) and optional profiling.
//! Move comments carry post-move evaluation, decision alternatives and optional
//! profiling. Truncated games have no RE. Generation/export timings finish
//! after annotation serialization, so those remain stderr-only.

pub mod parallel;

/// Bump when the direct random policy changes. Binary fingerprints additionally
/// isolate ratings across rules, search, RNG and inference implementation changes.
pub const REFERENCE_ID: &str = "random-no-eye-fill-v1";

use crate::{
    action::random_move,
    inference::{Evaluator, LeafEstimate},
    profile::{GameProfile, MoveProfile},
    search::{SearchConfig, Visit, color_sign, rollout::RolloutStats, search, search_neural},
    targets::{BOARD_AREA, normalized_score},
};
use continugo_analysis::GeometricAvailability;
use continugo_rules::{Board, Color, Game, Play, Stone, sgf, territory};
use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::Serialize;
use std::{cmp::Reverse, time::Instant};

#[derive(Clone, Copy, Debug, Serialize, clap::ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum Engine {
    Random,
    Mcts,
    Neural,
}

/// Randomly spend extra search on policy-training positions. The cheap budget
/// remains SearchConfig::simulations; this schedule never applies to evaluation.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct PolicySearchConfig {
    pub simulations: u32,
    pub probability: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct SelfplayConfig {
    pub engine: Engine,
    pub max_moves: usize,
    pub search: SearchConfig,
    pub policy_search: Option<PolicySearchConfig>,
}

impl SelfplayConfig {
    pub fn validate(&self) -> Result<(), String> {
        self.search.validate()?;
        if let Some(policy) = self.policy_search
            && (matches!(self.engine, Engine::Random)
                || self.search.simulations < 32
                || policy.simulations < self.search.simulations
                || !policy.probability.is_finite()
                || !(0.0 < policy.probability && policy.probability <= 1.0))
        {
            return Err("policy search requires a search engine, cheap simulations >= 32, policy simulations >= cheap simulations, and probability in (0,1]".into());
        }
        if self.search.bootstrap_avoid_eyes && !matches!(self.engine, Engine::Mcts) {
            return Err("bootstrap eye guidance requires --engine mcts".into());
        }
        if matches!(self.engine, Engine::Random) && self.search.rollout.is_some() {
            return Err("rollout leaf evaluation requires a search engine".into());
        }
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
    /// Whether each real visit distribution is eligible for policy supervision.
    pub policy_training: Vec<bool>,
    /// Post-move estimates in the mover's perspective, aligned with SGF moves.
    pub move_evaluations: Vec<MoveEvaluation>,
    /// None for a truncated game; otherwise the authoritative scored outcome.
    pub result: Option<territory::Outcome>,
    pub profile: Option<GameProfile>,
    pub model: Option<String>,
    pub opponent: Option<String>,
    pub evaluation: Option<String>,
    pub rollout_stats: RolloutStats,
}

#[derive(Serialize)]
struct GameComment<'a> {
    seed: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    search: Option<&'a SearchConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    policy_search: Option<&'a PolicySearchConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    profile: Option<&'a GameProfile>,
    #[serde(skip_serializing_if = "Option::is_none")]
    opponent: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    evaluation: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rollout_stats: Option<&'a RolloutStats>,
}

impl GameMetadata {
    fn comment(&self) -> GameComment<'_> {
        GameComment {
            seed: self.seed,
            search: (!matches!(self.config.engine, Engine::Random)).then_some(&self.config.search),
            policy_search: self.config.policy_search.as_ref(),
            model: self.model.as_deref(),
            profile: self.profile.as_ref(),
            opponent: self.opponent.as_deref(),
            evaluation: self.evaluation.as_deref(),
            rollout_stats: self
                .config
                .search
                .rollout
                .as_ref()
                .map(|_| &self.rollout_stats),
        }
    }
}

/// Display annotations only, never policy/outcome/score training labels.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct MoveEvaluation {
    /// Expected points percentage: draws count as half, not strict P(win).
    pub win_pct: Option<f64>,
    /// Expected final signed margin in SGF stone-area units, not raw board area.
    pub score: Option<f64>,
}

impl MoveEvaluation {
    fn after_move(game: &Game, evaluator: Option<&mut dyn Evaluator>) -> Result<Self, String> {
        let estimate = if game.is_finished() {
            let areas = territory::areas(game.position().stones());
            let sign = color_sign(game.on_move().other());
            LeafEstimate {
                value: match territory::result(areas) {
                    territory::Outcome::Draw => 0.0,
                    territory::Outcome::Win(color, _) => sign * color_sign(color),
                },
                score: sign * normalized_score(areas),
            }
        } else if let Some(evaluator) = evaluator {
            let estimate = evaluator.estimate(game)?;
            // The network sees the opponent now on turn, not the player who moved.
            LeafEstimate {
                value: -estimate.value,
                score: -estimate.score,
            }
        } else {
            return Ok(Self::default());
        };
        estimate.utility(0.0)?; // Reject invalid estimates, rather than emitting fake nulls.
        let stone_area = std::f64::consts::PI * f64::from(game.board().stone_radius()).powi(2);
        let rounded = |value: f64| {
            let value = (value * 10.0).round() / 10.0;
            if value == 0.0 { 0.0 } else { value } // Avoid a displayed negative zero.
        };
        Ok(Self {
            win_pct: Some(rounded(50.0 * (estimate.value + 1.0))),
            score: Some(rounded(estimate.score * BOARD_AREA / stone_area)),
        })
    }
}

#[derive(Serialize)]
struct MoveComment<'a> {
    #[serde(flatten)]
    evaluation: MoveEvaluation,
    #[serde(
        skip_serializing_if = "Option::is_none",
        serialize_with = "serialize_policy"
    )]
    policy: Option<&'a [Visit]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    train_policy: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    profile: Option<&'a MoveProfile>,
}

/// SGF comments store [coordinate string, visits] pairs, most visited first.
/// Sort only the serialized view; search order and move-selection RNG stay intact.
fn serialize_policy<S: serde::Serializer>(
    policy: &Option<&[Visit]>,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let Some(policy) = policy else {
        return serializer.serialize_none();
    };
    let mut ordered: Vec<_> = policy.iter().collect();
    ordered.sort_unstable_by_key(|visit| (Reverse(visit.visits), visit.action.index()));
    let pairs = ordered
        .into_iter()
        .map(|visit| {
            // Reuse the shared SGF codec; color is implicit in the move node.
            let play = match visit.action.spot() {
                Some(spot) => Play::Place(Stone {
                    color: Color::Black,
                    spot,
                }),
                None => Play::Pass(Color::Black),
            };
            let encoded = sgf::encode_play(play).map_err(serde::ser::Error::custom)?;
            Ok((encoded[2..encoded.len() - 1].to_owned(), visit.visits))
        })
        .collect::<Result<Vec<_>, S::Error>>()?;
    pairs.serialize(serializer)
}

#[derive(Debug)]
pub struct GameRecord {
    pub sgf: String,
    pub metadata: GameMetadata,
}

impl GameRecord {
    pub fn annotate_evaluation(&mut self, binary_id: &str) -> Result<(), String> {
        self.metadata.evaluation = Some(binary_id.into());
        let mut record = sgf::decode_record(&self.sgf)?;
        let comment = record
            .root
            .iter_mut()
            .find(|(name, _)| name == "C")
            .ok_or("generated game has no root comment")?;
        comment.1 =
            vec![serde_json::to_string(&self.metadata.comment()).map_err(|e| e.to_string())?];
        self.sgf = sgf::encode_record(&record)?;
        Ok(())
    }
}

pub fn generate(board: Board, config: &SelfplayConfig, seed: u64) -> Result<GameRecord, String> {
    generate_inner::<false>(board, config, seed, None, None, None)
}

pub fn generate_profiled(
    board: Board,
    config: &SelfplayConfig,
    seed: u64,
) -> Result<GameRecord, String> {
    if !matches!(config.engine, Engine::Random) {
        return Err("Detailed profiling currently supports --engine random only".into());
    }
    generate_inner::<true>(board, config, seed, None, None, None)
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
    generate_inner::<false>(board, config, seed, Some(evaluator), None, None)
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
    generate_inner::<false>(board, config, seed, Some(black), Some(white), None)
}

/// Neural maximum-visit play versus the unmodified direct random eye-avoiding
/// policy. Random moves have no fabricated MCTS targets.
pub fn generate_reference_match(
    board: Board,
    config: &SelfplayConfig,
    seed: u64,
    evaluator: &mut dyn Evaluator,
    candidate_black: bool,
) -> Result<GameRecord, String> {
    if !matches!(config.engine, Engine::Neural) {
        return Err("reference matches require the neural engine".into());
    }
    let reference_color = if candidate_black {
        Color::White
    } else {
        Color::Black
    };
    generate_inner::<false>(
        board,
        config,
        seed,
        Some(evaluator),
        None,
        Some(reference_color),
    )
}

/// Diagnostic matches for a network-independent bootstrap search. These are
/// evaluation SGFs, never eligible self-play data for the training supervisor.
pub fn generate_unguided_reference_match(
    board: Board,
    config: &SelfplayConfig,
    seed: u64,
    candidate_black: bool,
) -> Result<GameRecord, String> {
    if !matches!(config.engine, Engine::Mcts) {
        return Err("unguided reference matches require the MCTS engine".into());
    }
    let reference_color = if candidate_black {
        Color::White
    } else {
        Color::Black
    };
    generate_inner::<false>(board, config, seed, None, None, Some(reference_color))
}

fn generate_inner<const PROFILE: bool>(
    board: Board,
    config: &SelfplayConfig,
    seed: u64,
    mut evaluator: Option<&mut dyn Evaluator>,
    mut opponent: Option<&mut dyn Evaluator>,
    reference_color: Option<Color>,
) -> Result<GameRecord, String> {
    config.validate()?;
    if matches!(config.engine, Engine::Neural) && evaluator.is_none() {
        return Err("neural engine requires a loaded model".into());
    }
    let mut model = evaluator.as_ref().map(|e| e.model_id().to_owned());
    let mut opponent_id = opponent.as_ref().map(|e| e.model_id().to_owned());
    let player = match config.engine {
        Engine::Random => "random".to_owned(),
        Engine::Mcts => "mcts".to_owned(),
        Engine::Neural => format!("neural-{}", evaluator.as_ref().unwrap().network_name()),
    };
    let mut players = [
        player.clone(),
        opponent
            .as_ref()
            .map_or(player, |e| format!("neural-{}", e.network_name())),
    ];
    if let Some(color) = reference_color {
        if color == Color::Black {
            opponent_id = model.take();
            model = Some(REFERENCE_ID.into());
            players[0] = REFERENCE_ID.into();
        } else {
            opponent_id = Some(REFERENCE_ID.into());
            players[1] = REFERENCE_ID.into();
        }
    }
    let evaluation_game = opponent.is_some() || reference_color.is_some();
    if evaluation_game && config.search.bootstrap_avoid_eyes {
        return Err("bootstrap eye guidance is forbidden in evaluation games".into());
    }
    if evaluation_game && config.policy_search.is_some() {
        return Err("randomized policy search is forbidden in evaluation games".into());
    }
    let started = PROFILE.then(Instant::now);
    let mut profile = GameProfile::default();
    let mut rng = StdRng::seed_from_u64(seed);
    // Separate stream: cap selection is independent of search/move RNG consumption
    // and of cross-game batching. Disabled schedules consume no extra search RNG.
    let mut cap_rng = StdRng::seed_from_u64(seed ^ 0x706f_6c69_6379_6361);
    let mut game = Game::new(board);
    let index_started = PROFILE.then(Instant::now);
    let mut availability = (matches!(config.engine, Engine::Random) || reference_color.is_some())
        .then(|| GeometricAvailability::new(game.position()));
    if let Some(started) = index_started {
        profile.availability_init_seconds = started.elapsed().as_secs_f64();
    }
    let mut policies = Vec::new();
    let mut policy_training = Vec::new();
    let mut move_evaluations = Vec::new();
    let mut rollout_stats = RolloutStats::default();
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
        let random_turn =
            matches!(config.engine, Engine::Random) || reference_color == Some(game.on_move());
        // Decide before inspecting/searching the position, never from its result.
        let train_policy = !random_turn
            && config
                .policy_search
                .is_none_or(|policy| cap_rng.random_bool(policy.probability));
        let mut search_config = config.search.clone();
        if train_policy && let Some(policy) = config.policy_search {
            search_config.simulations = policy.simulations;
        }
        let (next, policy) = if random_turn {
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
        } else {
            match config.engine {
                Engine::Random => unreachable!(),
                Engine::Mcts => {
                    let result = search(&game, &search_config, &mut rng)?;
                    rollout_stats.merge(&result.rollout_stats);
                    let action = if evaluation_game {
                        result.best(&mut rng)
                    } else {
                        result.sample(&mut rng)
                    };
                    (action.apply(&game)?, Some(result.policy))
                }
                Engine::Neural => {
                    let result = search_neural(
                        &game,
                        &search_config,
                        if opponent.is_some() && game.on_move() == Color::White {
                            opponent.as_deref_mut().expect("opposing evaluator")
                        } else {
                            evaluator.as_deref_mut().expect("validated evaluator")
                        },
                        &mut rng,
                    )?;
                    rollout_stats.merge(&result.rollout_stats);
                    let action = if evaluation_game {
                        result.best(&mut rng)
                    } else {
                        result.sample(&mut rng)
                    };
                    (action.apply(&game)?, Some(result.policy))
                }
            }
        };
        let evaluation = if random_turn || !matches!(config.engine, Engine::Neural) {
            MoveEvaluation::after_move(&next, None)?
        } else if opponent.is_some() && game.on_move() == Color::White {
            MoveEvaluation::after_move(
                &next,
                Some(opponent.as_deref_mut().expect("opposing evaluator")),
            )?
        } else {
            MoveEvaluation::after_move(
                &next,
                Some(evaluator.as_deref_mut().expect("validated evaluator")),
            )?
        };
        move_evaluations.push(evaluation);
        if let Some(index) = &mut availability {
            let update_started = PROFILE.then(Instant::now);
            index.sync(next.position());
            if let Some(started) = update_started {
                move_profile.availability_update.record(started);
            }
        }
        game = next;
        policies.push(policy);
        policy_training.push(train_policy);
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
        policy_training,
        move_evaluations,
        result,
        profile: PROFILE.then_some(profile),
        model,
        opponent: opponent_id,
        evaluation: None,
        rollout_stats,
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
    record.root.extend([
        ("PB".into(), vec![players[0].clone()]),
        ("PW".into(), vec![players[1].clone()]),
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
            evaluation: metadata.move_evaluations[index],
            policy: policy.as_deref(),
            train_policy: metadata
                .config
                .policy_search
                .map(|_| metadata.policy_training[index]),
            profile: metadata.profile.as_ref().map(|p| &p.moves[index]),
        };
        record.move_properties.push(vec![(
            "C".into(),
            vec![serde_json::to_string(&comment).map_err(|e| e.to_string())?],
        )]);
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
    fn policy_comments_use_sgf_pairs_sorted_by_visits_without_changing_search_order() {
        use crate::action::Action;
        let policy: Vec<_> = [
            (200_200, 3),
            (1_000_000, 5),
            (38_038, 3),
            (999_999, 0),
            (0, 0),
        ]
        .into_iter()
        .map(|(index, visits)| Visit {
            action: Action::from_index(index).unwrap(),
            visits,
        })
        .collect();
        let comment = serde_json::to_string(&MoveComment {
            evaluation: MoveEvaluation {
                win_pct: Some(63.4),
                score: Some(5.2),
            },
            policy: Some(&policy),
            train_policy: None,
            profile: None,
        })
        .unwrap();
        assert_eq!(
            comment,
            r#"{"win_pct":63.4,"score":5.2,"policy":[["",5],["aMaM",3],["dSdS",3],["aaaa",0],["tltl",0]]}"#
        );
        assert_eq!(policy[0].action.index(), 200_200);
        assert_eq!(policy[1].action, Action::PASS);
        let mut record = sgf::Record {
            plays: vec![Play::Pass(Color::Black)],
            ..Default::default()
        };
        record.move_properties = vec![vec![("C".into(), vec![comment.clone()])]];
        let text = sgf::encode_record(&record).unwrap();
        assert!(text.contains(r#"["",5\]"#));
        assert_eq!(
            sgf::decode_record(&text).unwrap().move_properties[0][0].1[0],
            comment
        );
    }

    struct FixedEstimate {
        value: f64,
        score: f64,
        seen: Vec<Game>,
    }

    impl Evaluator for FixedEstimate {
        fn model_id(&self) -> &str {
            "fixed-estimate"
        }
        fn evaluate(&mut self, game: &Game) -> Result<crate::inference::Evaluation, String> {
            // Exactly one placement proposal makes one-simulation match moves
            // deterministic, independently of the mocked value/score heads.
            let index = 200_200 + 150 * game.record().len() as u32;
            let mut logits = vec![f32::NEG_INFINITY; 1_000_001];
            let mut legal = vec![0; 1_000_000];
            logits[index as usize] = 0.0;
            legal[index as usize] = 1;
            logits[1_000_000] = -100.0;
            Ok(crate::inference::Evaluation {
                value: self.value,
                score: self.score,
                logits,
                legal,
            })
        }
        fn estimate(&mut self, game: &Game) -> Result<LeafEstimate, String> {
            assert!(
                !game.is_finished(),
                "terminal annotations must bypass inference"
            );
            self.seen.push(game.clone());
            Ok(LeafEstimate {
                value: self.value,
                score: self.score,
            })
        }
    }

    #[test]
    fn post_move_annotations_flip_both_heads_and_convert_physical_score_units() {
        use continugo_rules::Spot;
        for (size, expected_score) in [(9, 10.5), (13, 22.0), (19, 47.1)] {
            let after_black = Game::new(Board::new(size).unwrap())
                .play(Spot { x: 500, y: 500 })
                .unwrap();
            let after_white = after_black.pass().unwrap();
            for game in [after_black, after_white] {
                let mut evaluator = FixedEstimate {
                    value: -0.268,
                    score: -0.1,
                    seen: Vec::new(),
                };
                let annotation = MoveEvaluation::after_move(&game, Some(&mut evaluator)).unwrap();
                assert_eq!(evaluator.seen, vec![game]);
                assert_eq!(annotation.win_pct, Some(63.4));
                assert_eq!(annotation.score, Some(expected_score));
            }
        }
    }

    #[test]
    fn move_annotations_are_unknown_without_a_model_and_exact_at_terminal_positions() {
        use continugo_rules::Spot;
        let draw = Game::default().pass().unwrap().pass().unwrap();
        let black_win = Game::default()
            .play(Spot { x: 500, y: 500 })
            .unwrap()
            .pass()
            .unwrap()
            .pass()
            .unwrap();
        let white_win = Game::default()
            .pass()
            .unwrap()
            .play(Spot { x: 500, y: 500 })
            .unwrap()
            .pass()
            .unwrap()
            .pass()
            .unwrap();
        let white_loss = Game::default()
            .play(Spot { x: 500, y: 500 })
            .unwrap()
            .play(Spot { x: 900, y: 900 })
            .unwrap()
            .pass()
            .unwrap()
            .pass()
            .unwrap();
        let unknown = MoveEvaluation::after_move(&Game::default().pass().unwrap(), None).unwrap();
        assert_eq!(
            serde_json::to_string(&unknown).unwrap(),
            r#"{"win_pct":null,"score":null}"#
        );
        let mut evaluator = FixedEstimate {
            value: f64::NAN,
            score: f64::NAN,
            seen: Vec::new(),
        };
        for (game, win_pct) in [
            (draw, 50.0),
            (black_win, 100.0),
            (white_win, 100.0),
            (white_loss, 0.0),
        ] {
            let annotation = MoveEvaluation::after_move(&game, Some(&mut evaluator)).unwrap();
            assert_eq!(annotation.win_pct, Some(win_pct));
            let result = territory::result(territory::areas(game.position().stones()));
            let expected_score = match result {
                territory::Outcome::Draw => 0.0,
                territory::Outcome::Win(winner, _) => {
                    let result = sgf::format_result(game.board(), result);
                    let margin: f64 = result.split_once('+').unwrap().1.parse().unwrap();
                    if winner == game.on_move().other() {
                        margin
                    } else {
                        -margin
                    }
                }
            };
            assert_eq!(annotation.score, Some(expected_score));
        }
        assert!(evaluator.seen.is_empty());
    }

    #[test]
    fn terminal_annotations_accept_polygon_area_roundoff_without_a_network() {
        use continugo_rules::Spot;
        let mut rng = StdRng::seed_from_u64(0);
        let mut overshot = false;
        for _ in 0..128 {
            let mut game = Game::new(Board::new(9).unwrap());
            for _ in 0..3 {
                let spot = Spot {
                    x: rng.random_range(100..900),
                    y: rng.random_range(100..900),
                };
                if let Ok(next) = game.play(spot) {
                    game = next.pass().unwrap();
                }
            }
            if game.record().is_empty() {
                continue;
            }
            game = game.pass().unwrap();
            assert!(game.is_finished());
            let areas = territory::areas(game.position().stones());
            overshot |= areas.black > BOARD_AREA;
            let annotation = MoveEvaluation::after_move(&game, None).unwrap();
            assert_eq!(annotation.win_pct, Some(100.0));
            let stone_area = std::f64::consts::PI * f64::from(game.board().stone_radius()).powi(2);
            let expected = (BOARD_AREA / stone_area * 10.0).round() / 10.0;
            assert_eq!(annotation.score, Some(expected));
        }
        assert!(
            overshot,
            "regression must exercise genuine polygon accumulation overshoot"
        );
    }

    #[test]
    fn invalid_annotation_estimates_are_errors_not_unknown_values() {
        let game = Game::default().pass().unwrap();
        for (value, score) in [
            (f64::NAN, 0.0),
            (0.0, f64::INFINITY),
            (1.1, 0.0),
            (0.0, -1.1),
            (f64::from_bits(1.0f64.to_bits() + 1), 0.0),
            (0.0, -f64::from_bits(1.0f64.to_bits() + 1)),
        ] {
            let mut evaluator = FixedEstimate {
                value,
                score,
                seen: Vec::new(),
            };
            assert!(MoveEvaluation::after_move(&game, Some(&mut evaluator)).is_err());
        }
    }

    #[test]
    fn match_annotations_use_the_movers_network_not_the_next_players_network() {
        let mut black = FixedEstimate {
            value: 0.268,
            score: 0.1,
            seen: Vec::new(),
        };
        let mut white = FixedEstimate {
            value: -0.268,
            score: -0.1,
            seen: Vec::new(),
        };
        let config = SelfplayConfig {
            engine: Engine::Neural,
            policy_search: None,
            max_moves: 4,
            search: SearchConfig {
                simulations: 1,
                ..Default::default()
            },
        };
        let record = generate_match(Board::default(), &config, 7, &mut black, &mut white).unwrap();
        let decoded = sgf::decode_record(&record.sgf).unwrap();
        assert_eq!(decoded.plays.len(), 4);
        assert!(record.metadata.result.is_none());
        for (ply, properties) in decoded.move_properties.iter().enumerate() {
            let comment = &properties.iter().find(|(key, _)| key == "C").unwrap().1[0];
            let value: serde_json::Value = serde_json::from_str(comment).unwrap();
            assert!(comment.starts_with("{\"win_pct\":"));
            assert!(comment.find("\"score\"").unwrap() < comment.find("\"policy\"").unwrap());
            assert_eq!(value["win_pct"], if ply % 2 == 0 { 36.6 } else { 63.4 });
            assert_eq!(value["score"], if ply % 2 == 0 { -22.0 } else { 22.0 });
        }
        replay(&record);
    }

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
                    score: 0.0,
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
            policy_search: None,
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
        for candidate_black in [true, false] {
            let before = black.calls;
            let mut reference = generate_reference_match(
                Board::new(9).unwrap(),
                &config,
                7,
                &mut black,
                candidate_black,
            )
            .unwrap();
            assert!(black.calls > before);
            assert!(reference.metadata.result.is_none());
            reference.annotate_evaluation("test-fingerprint").unwrap();
            let info = crate::replay::ReplayGame::load(&reference.sgf)
                .unwrap()
                .info(4)
                .unwrap();
            assert_eq!(
                info.policy_visit_totals,
                (0..8)
                    .map(|ply| {
                        if (ply % 2 == 0) == candidate_black {
                            Some(4)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            );
            assert_eq!(info.root_comment.unwrap()["evaluation"], "test-fingerprint");
            assert_eq!(
                reference.metadata.model.as_deref(),
                Some(if candidate_black {
                    "black"
                } else {
                    REFERENCE_ID
                })
            );
            assert_eq!(
                reference.metadata.opponent.as_deref(),
                Some(if candidate_black {
                    REFERENCE_ID
                } else {
                    "black"
                })
            );
            let restored = replay(&reference);
            for (ply, play) in restored.record().iter().enumerate() {
                if (ply % 2 == 0) != candidate_black {
                    assert!(
                        matches!(play, continugo_rules::Play::Place(_)),
                        "random bot passed prematurely"
                    );
                }
            }
        }
        let complete = generate_reference_match(
            Board::new(9).unwrap(),
            &SelfplayConfig {
                max_moves: 512,
                search: SearchConfig {
                    simulations: 1,
                    ..config.search
                },
                ..config
            },
            7,
            &mut black,
            true,
        )
        .unwrap();
        assert!(matches!(
            complete.metadata.result,
            Some(territory::Outcome::Win(Color::White, _))
        ));
        assert!(replay(&complete).is_finished());
    }
    use continugo_analysis::fills_small_eye;

    #[test]
    fn guided_bootstrap_has_real_visits_natural_results_and_no_noncap_eye_targets() {
        let config = SelfplayConfig {
            engine: Engine::Mcts,
            policy_search: None,
            max_moves: 1024,
            search: SearchConfig {
                simulations: 8,
                bootstrap_avoid_eyes: true,
                ..SearchConfig::default()
            },
        };
        let record = generate(Board::new(9).unwrap(), &config, 900000).unwrap();
        let restored = replay(&record);
        assert!(restored.is_finished());
        assert!(record.metadata.result.is_some());
        let mut game = Game::new(restored.board());
        for (play, policy) in restored.record().iter().zip(&record.metadata.policies) {
            let policy = policy.as_ref().unwrap();
            assert_eq!(policy.iter().map(|v| v.visits).sum::<u32>(), 8);
            for entry in policy {
                if let Some(spot) = entry.action.spot() {
                    let next = game.play(spot).unwrap();
                    assert!(
                        next.position().stones().len() < game.position().stones().len() + 1
                            || !fills_small_eye(game.position(), game.on_move(), spot)
                    );
                }
            }
            game = match play {
                Play::Place(stone) => game.play(stone.spot).unwrap(),
                Play::Pass(_) => game.pass().unwrap(),
            };
        }
        assert!(
            generate_unguided_reference_match(restored.board(), &config, 1, true)
                .unwrap_err()
                .contains("forbidden in evaluation")
        );
    }

    #[test]
    fn randomized_caps_keep_real_visits_and_value_labels_but_mark_policy_positions() {
        struct PassOnly;
        impl Evaluator for PassOnly {
            fn model_id(&self) -> &str {
                "pass-only"
            }
            fn evaluate(&mut self, _: &Game) -> Result<crate::inference::Evaluation, String> {
                let mut logits = vec![f32::NEG_INFINITY; 1_000_001];
                logits[1_000_000] = 0.0;
                Ok(crate::inference::Evaluation {
                    value: 0.0,
                    score: 0.0,
                    logits,
                    legal: vec![0; 1_000_000],
                })
            }
        }
        let mut config = config(Engine::Neural);
        config.search.simulations = 32;
        config.policy_search = Some(PolicySearchConfig {
            simulations: 256,
            probability: 0.25,
        });
        let mut counts = [0; 2];
        for seed in 0..16 {
            let record =
                generate_neural(Board::new(9).unwrap(), &config, seed, &mut PassOnly).unwrap();
            let again =
                generate_neural(Board::new(9).unwrap(), &config, seed, &mut PassOnly).unwrap();
            assert_eq!(record.sgf, again.sgf);
            let loaded = crate::replay::ReplayGame::load(&record.sgf).unwrap();
            let info = loaded.info(4).unwrap();
            assert!(info.labeled);
            assert_eq!(info.eligible_plies, vec![0, 1]);
            assert_eq!(
                info.root_comment.as_ref().unwrap()["policy_search"]["simulations"],
                256
            );
            for (ply, &enabled) in record.metadata.policy_training.iter().enumerate() {
                counts[usize::from(enabled)] += 1;
                assert_eq!(
                    info.policy_visit_totals[ply],
                    Some(if enabled { 256 } else { 32 })
                );
                assert_eq!(info.policy_training_plies.contains(&ply), enabled);
                assert_eq!(loaded.position(ply).unwrap().outcome, Some(0.0));
            }
            replay(&record);
        }
        assert!(counts[0] > 0 && counts[1] > 0);
        assert!(
            generate_match(
                Board::new(9).unwrap(),
                &config,
                0,
                &mut PassOnly,
                &mut PassOnly
            )
            .unwrap_err()
            .contains("forbidden in evaluation")
        );
        for (cheap, expensive, probability) in [
            (8, 256, 0.25),
            (64, 32, 0.25),
            (32, 256, f64::NAN),
            (32, 256, 1.1),
        ] {
            config.search.simulations = cheap;
            config.policy_search = Some(PolicySearchConfig {
                simulations: expensive,
                probability,
            });
            assert!(config.validate().is_err());
        }
    }

    fn config(engine: Engine) -> SelfplayConfig {
        SelfplayConfig {
            engine,
            policy_search: None,
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
        assert_eq!(plays.len(), metadata.move_evaluations.len());
        for (index, (play, policy)) in plays.iter().zip(&metadata.policies).enumerate() {
            let comment = MoveComment {
                evaluation: metadata.move_evaluations[index],
                policy: policy.as_deref(),
                train_policy: metadata
                    .config
                    .policy_search
                    .map(|_| metadata.policy_training[index]),
                profile: metadata.profile.as_ref().map(|p| &p.moves[index]),
            };
            assert_eq!(
                decoded.move_properties[index],
                vec![("C".into(), vec![serde_json::to_string(&comment).unwrap()])]
            );
            let next = match play {
                Play::Place(stone) => game.play(stone.spot),
                Play::Pass(_) => game.pass(),
            }
            .unwrap();
            let random_turn = matches!(metadata.config.engine, Engine::Random)
                || (if game.on_move() == Color::Black {
                    &metadata.model
                } else {
                    &metadata.opponent
                })
                .as_deref()
                    == Some(REFERENCE_ID);
            if random_turn && let Play::Place(stone) = play {
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
                    if metadata.policy_training[index] {
                        metadata
                            .config
                            .policy_search
                            .map_or(metadata.config.search.simulations, |p| p.simulations)
                    } else {
                        metadata.config.search.simulations
                    }
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
