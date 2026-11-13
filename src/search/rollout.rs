//! Optional cold-start leaf evaluation from naturally completed legal playouts.
//! No unfinished-board scoring, eye filter, neural value or fabricated cap draws.

use super::{SearchValue, color_sign, terminal_estimate};
use crate::action::Proposals;
use continugo_rules::Game;
use rand::Rng;
use serde::Serialize;

/// Standalone self-play/arena controls. The gated supervisor intentionally does
/// not expose this experimental evaluator until its strength/censoring is tested.
#[derive(clap::Args)]
pub struct RolloutArgs {
    /// Completed-game playouts per leaf; zero keeps normal neutral/neural values.
    #[arg(long, default_value_t = 0)]
    pub rollouts: u32,
    /// Safety cap per hypothetical playout, not an outcome-scoring rule.
    #[arg(long, default_value_t = 512)]
    pub rollout_max_moves: usize,
    /// Random legal pass probability, independent of whether a pass would win.
    #[arg(long, default_value_t = 1.0 / 32.0)]
    pub rollout_pass_probability: f64,
}

impl RolloutArgs {
    pub fn config(&self) -> Option<RolloutConfig> {
        (self.rollouts > 0).then_some(RolloutConfig {
            playouts: self.rollouts,
            max_moves: self.rollout_max_moves,
            pass_probability: self.rollout_pass_probability,
        })
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct RolloutConfig {
    pub playouts: u32,
    pub max_moves: usize,
    /// Otherwise try a bounded uniform-domain legal placement; failure also passes.
    pub pass_probability: f64,
}

impl RolloutConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.playouts == 0
            || self.max_moves == 0
            || !self.pass_probability.is_finite()
            || !(0.0..=1.0).contains(&self.pass_probability)
        {
            return Err(
                "rollouts need positive playout/move limits and pass probability in 0..1".into(),
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct RolloutStats {
    pub completed: u64,
    pub truncated: u64,
    pub wins: u64,
    pub losses: u64,
    pub draws: u64,
    pub plies: u64,
}

impl RolloutStats {
    pub fn merge(&mut self, other: &Self) {
        self.completed += other.completed;
        self.truncated += other.truncated;
        self.wins += other.wins;
        self.losses += other.losses;
        self.draws += other.draws;
        self.plies += other.plies;
    }
}

/// Returns None if there is no completed evidence. The search must not count
/// unknown continuations as draws or as observed zero-valued outcomes.
pub(super) fn evaluate(
    source: &Game,
    config: &RolloutConfig,
    proposal_attempts: usize,
    score_weight: f64,
    rng: &mut impl Rng,
    stats: &mut RolloutStats,
) -> Option<SearchValue> {
    let perspective = color_sign(source.on_move());
    let mut total = SearchValue::default();
    let mut completed = 0;
    for _ in 0..config.playouts {
        let mut game = source.clone();
        for _ in 0..config.max_moves {
            if game.is_finished() {
                break;
            }
            let pass = game.pass().expect("unfinished rollout");
            // A small win is not a reason to stop maximizing score.
            game = if rng.random_bool(config.pass_probability) {
                pass
            } else {
                let prepared = game.prepare();
                Proposals::new(rng)
                    .legal(&prepared, proposal_attempts)
                    .map_or(pass, |(_, next)| next)
            };
            stats.plies += 1;
        }
        if game.is_finished() {
            let (estimate, jigo) = terminal_estimate(&game);
            let sign = color_sign(game.on_move()) * perspective;
            let value = if jigo { 0.0 } else { sign * estimate.value };
            stats.completed += 1;
            stats.wins += u64::from(value > 0.0);
            stats.losses += u64::from(value < 0.0);
            stats.draws += u64::from(value == 0.0);
            let mut evidence = SearchValue {
                value: estimate
                    .utility(score_weight)
                    .expect("validated score weight"),
                jigo_penalty: if jigo { 1.0 - score_weight } else { 0.0 },
            };
            if sign < 0.0 {
                evidence = evidence.flipped();
            }
            total.value += evidence.value;
            total.jigo_penalty += evidence.jigo_penalty;
            completed += 1;
        } else {
            stats.truncated += 1;
        }
    }
    (completed > 0).then(|| SearchValue {
        value: total.value / f64::from(completed),
        jigo_penalty: total.jigo_penalty / f64::from(completed),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use continugo_rules::{Board, Spot};
    use rand::{SeedableRng, rngs::StdRng};

    fn config(max_moves: usize) -> RolloutConfig {
        RolloutConfig {
            playouts: 4,
            max_moves,
            pass_probability: 1.0,
        }
    }

    #[test]
    fn capped_continuations_are_unknown_not_draws() {
        let mut stats = RolloutStats::default();
        let value = evaluate(
            &Game::default(),
            &config(1),
            128,
            1.0,
            &mut StdRng::seed_from_u64(0),
            &mut stats,
        );
        assert_eq!(value, None);
        assert_eq!(stats.completed, 0);
        assert_eq!(stats.draws, 0);
        assert_eq!(stats.truncated, 4);
    }

    #[test]
    fn terminal_outcomes_and_player_perspectives_are_real() {
        let mut stats = RolloutStats::default();
        let game = Game::new(Board::new(9).unwrap())
            .play(Spot { x: 500, y: 500 })
            .unwrap();
        let mut rng = StdRng::seed_from_u64(0);
        assert_eq!(
            evaluate(&game, &config(2), 128, 1.0, &mut rng, &mut stats).map(|v| v.value),
            Some(-1.0)
        );
        assert_eq!(stats.losses, 4);
        assert_eq!(stats.draws, 0);
        assert_eq!(
            evaluate(
                &game.pass().unwrap(),
                &config(1),
                128,
                1.0,
                &mut rng,
                &mut stats
            )
            .map(|v| v.value),
            Some(1.0)
        );
        assert_eq!(stats.wins, 4);
        assert_eq!(
            evaluate(&Game::default(), &config(2), 128, 1.0, &mut rng, &mut stats).map(|v| v.value),
            Some(0.0)
        );
        assert_eq!(stats.draws, 4);
    }

    #[test]
    fn jigo_rollouts_penalize_both_players_but_statistics_still_report_draws() {
        for game in [Game::default(), Game::default().pass().unwrap()] {
            let mut stats = RolloutStats::default();
            let value = evaluate(
                &game,
                &config(2),
                128,
                0.5,
                &mut StdRng::seed_from_u64(0),
                &mut stats,
            )
            .unwrap();
            assert_eq!(value.value, -0.5);
            assert_eq!(value.flipped().value, -0.5);
            assert_eq!(value.jigo_penalty, 0.5);
            assert_eq!(stats.draws, 4);
            assert_eq!(stats.losses, 0);
            assert_eq!(stats.wins, 0);
        }
    }

    #[test]
    fn winning_pass_does_not_override_the_rollout_policy() {
        let game = Game::default()
            .play(Spot { x: 500, y: 500 })
            .unwrap()
            .pass()
            .unwrap();
        let cfg = RolloutConfig {
            pass_probability: 0.0,
            ..config(1)
        };
        let mut stats = RolloutStats::default();
        let value = evaluate(
            &game,
            &cfg,
            128,
            1.0,
            &mut StdRng::seed_from_u64(0),
            &mut stats,
        );
        assert_eq!(value, None);
        assert_eq!(stats.completed, 0);
        assert_eq!(stats.truncated, 4);
        assert_eq!(stats.plies, 4);
    }

    #[test]
    fn completed_rollouts_use_margin_not_just_win_loss() {
        let game = Game::default()
            .play(Spot { x: 500, y: 500 })
            .unwrap()
            .play(Spot { x: 900, y: 900 })
            .unwrap()
            .play(Spot { x: 350, y: 350 })
            .unwrap()
            .pass()
            .unwrap();
        let expected = -terminal_estimate(&game.pass().unwrap()).0.score;
        assert!(expected > 0.0 && expected < 1.0);
        for (weight, expected) in [(1.0, expected), (0.0, 1.0)] {
            let mut stats = RolloutStats::default();
            let value = evaluate(
                &game,
                &config(1),
                128,
                weight,
                &mut StdRng::seed_from_u64(0),
                &mut stats,
            );
            assert_eq!(value.map(|v| v.value), Some(expected));
            assert_eq!(stats.wins, 4);
        }
    }

    #[test]
    fn invalid_configuration_is_rejected() {
        for cfg in [
            RolloutConfig {
                playouts: 0,
                ..config(1)
            },
            config(0),
            RolloutConfig {
                pass_probability: f64::NAN,
                ..config(1)
            },
            RolloutConfig {
                pass_probability: -0.1,
                ..config(1)
            },
        ] {
            assert!(cfg.validate().is_err());
        }
    }
}
