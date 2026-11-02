//! Optional cold-start leaf evaluation from naturally completed legal playouts.
//! No unfinished-board scoring, eye filter, neural value or fabricated cap draws.

use super::{black_outcome, color_sign};
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
    /// Random legal pass probability; immediate winning passes are taken exactly.
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
    rng: &mut impl Rng,
    stats: &mut RolloutStats,
) -> Option<f64> {
    let perspective = color_sign(source.on_move());
    let mut total = 0.0;
    let mut completed = 0;
    for _ in 0..config.playouts {
        let mut game = source.clone();
        for _ in 0..config.max_moves {
            if game.is_finished() {
                break;
            }
            let pass = game.pass().expect("unfinished rollout");
            // Take an actual immediate win, not a heuristic score of an
            // unfinished position. Other legal passes remain available.
            let wins_by_pass =
                pass.is_finished() && black_outcome(&pass) * color_sign(game.on_move()) == 1.0;
            game = if wins_by_pass || rng.random_bool(config.pass_probability) {
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
            let value = black_outcome(&game) * perspective;
            stats.completed += 1;
            stats.wins += u64::from(value > 0.0);
            stats.losses += u64::from(value < 0.0);
            stats.draws += u64::from(value == 0.0);
            total += value;
            completed += 1;
        } else {
            stats.truncated += 1;
        }
    }
    (completed > 0).then(|| total / f64::from(completed))
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
            evaluate(&game, &config(2), 128, &mut rng, &mut stats),
            Some(-1.0)
        );
        assert_eq!(stats.losses, 4);
        assert_eq!(stats.draws, 0);
        assert_eq!(
            evaluate(&game.pass().unwrap(), &config(1), 128, &mut rng, &mut stats),
            Some(1.0)
        );
        assert_eq!(stats.wins, 4);
        assert_eq!(
            evaluate(&Game::default(), &config(2), 128, &mut rng, &mut stats),
            Some(0.0)
        );
        assert_eq!(stats.draws, 4);
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
