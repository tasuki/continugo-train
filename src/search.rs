use crate::action::{Action, Proposals};
use continugo_rules::{Color, Game, territory};
use rand::Rng;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct SearchConfig {
    pub simulations: u32,
    pub cpuct: f64,
    pub widening: f64,
    pub widening_exponent: f64,
    pub proposal_attempts: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            simulations: 64,
            cpuct: 1.5,
            widening: 2.0,
            widening_exponent: 0.5,
            proposal_attempts: 128,
        }
    }
}

impl SearchConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.simulations == 0
            || !self.cpuct.is_finite()
            || self.cpuct <= 0.0
            || !self.widening.is_finite()
            || self.widening <= 0.0
            || !self.widening_exponent.is_finite()
            || !(0.0..1.0).contains(&self.widening_exponent)
            || self.widening_exponent == 0.0
            || self.proposal_attempts == 0
        {
            return Err("Search needs positive simulations, cpuct, widening, proposal attempts, and 0 < exponent < 1".into());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Visit {
    #[serde(serialize_with = "crate::action::serialize")]
    pub action: Action,
    pub visits: u32,
}

#[derive(Debug)]
pub struct SearchResult {
    pub policy: Vec<Visit>,
}

impl SearchResult {
    /// Temperature 1: sample exactly in proportion to root visit counts.
    pub fn sample(&self, rng: &mut impl Rng) -> Action {
        let total: u32 = self.policy.iter().map(|entry| entry.visits).sum();
        let mut ticket = rng.random_range(0..total);
        for entry in &self.policy {
            if ticket < entry.visits {
                return entry.action;
            }
            ticket -= entry.visits;
        }
        unreachable!("nonempty search policy")
    }
}

struct Edge {
    action: Action,
    child: Node,
}

struct Node {
    game: Game,
    visits: u32,
    // Always from this node's side-to-move perspective, not its parent's.
    value_sum: f64,
    proposals: Proposals,
    children: Vec<Edge>,
}

impl Node {
    fn new(game: Game, rng: &mut impl Rng) -> Self {
        Self {
            game,
            visits: 0,
            value_sum: 0.0,
            proposals: Proposals::new(rng),
            children: Vec::new(),
        }
    }

    fn update(&mut self, value: f64) -> f64 {
        self.visits += 1;
        self.value_sum += value;
        value
    }

    fn mean(&self) -> f64 {
        if self.visits == 0 {
            0.0
        } else {
            self.value_sum / f64::from(self.visits)
        }
    }

    fn simulate(&mut self, config: &SearchConfig, rng: &mut impl Rng) -> f64 {
        if self.game.is_finished() {
            let value = black_outcome(&self.game) * color_sign(self.game.on_move());
            return self.update(value);
        }
        if self.visits == 0 {
            // Bootstrap evaluator: neutral value, no handcrafted positional score
            // or rollout bot. Replace with network evaluation in the next stage.
            return self.update(0.0);
        }
        if self.children.is_empty() {
            self.children.push(Edge {
                action: Action::PASS,
                child: Node::new(self.game.pass().expect("unfinished game"), rng),
            });
        }
        let placement_limit = (config.widening
            * f64::from(self.visits).powf(config.widening_exponent))
        .ceil() as usize;
        if self.children.len() - 1 < placement_limit
            && let Some((action, game)) = self.proposals.legal(&self.game, config.proposal_attempts)
        {
            // At most one new coordinate per simulation. Rejected coordinates
            // consume proposal work, but never become actions in the tree.
            self.children.push(Edge {
                action,
                child: Node::new(game, rng),
            });
        }
        // Uniform prior conditional on the admitted candidates (including pass).
        // This is a baseline, not a full-million-action neural policy.
        let exploration = config.cpuct * f64::from(self.visits).sqrt() / self.children.len() as f64;
        let best = self
            .children
            .iter()
            .enumerate()
            .map(|(index, edge)| {
                let score = -edge.child.mean() + exploration / (1.0 + f64::from(edge.child.visits));
                (index, score, rng.random::<u64>())
            })
            .max_by(|a, b| a.1.total_cmp(&b.1).then(a.2.cmp(&b.2)))
            .unwrap()
            .0;
        let value = -self.children[best].child.simulate(config, rng);
        self.update(value)
    }
}

pub fn color_sign(color: Color) -> f64 {
    match color {
        Color::Black => 1.0,
        Color::White => -1.0,
    }
}

/// The caller must only use this as a target for naturally completed games.
pub fn black_outcome(game: &Game) -> f64 {
    match territory::result(territory::areas(game.position().stones())) {
        territory::Outcome::Draw => 0.0,
        territory::Outcome::Win(color, _) => color_sign(color),
    }
}

pub fn search(
    game: &Game,
    config: &SearchConfig,
    rng: &mut impl Rng,
) -> Result<SearchResult, String> {
    config.validate()?;
    if game.is_finished() {
        return Err("Cannot search a finished game".into());
    }
    let mut root = Node::new(game.clone(), rng);
    root.update(0.0); // Initial neutral evaluation; not a counted simulation.
    for _ in 0..config.simulations {
        root.simulate(config, rng);
    }
    Ok(SearchResult {
        policy: root
            .children
            .into_iter()
            .map(|edge| Visit {
                action: edge.action,
                visits: edge.child.visits,
            })
            .collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use continugo_rules::Spot;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn counts_are_conserved_and_candidates_are_legal_unique_and_widened() {
        let game = Game::default();
        let config = SearchConfig::default();
        let result = search(&game, &config, &mut StdRng::seed_from_u64(7)).unwrap();
        assert_eq!(
            result.policy.iter().map(|v| v.visits).sum::<u32>(),
            config.simulations
        );
        assert!(result.policy.len() > 2);
        assert!(
            result.policy.len()
                <= 1 + (config.widening
                    * f64::from(config.simulations).powf(config.widening_exponent))
                .ceil() as usize
        );
        for (index, entry) in result.policy.iter().enumerate() {
            assert!(entry.action.apply(&game).is_ok());
            assert!(
                !result.policy[..index]
                    .iter()
                    .any(|other| other.action == entry.action)
            );
        }
    }

    #[test]
    fn terminal_values_and_backup_have_the_correct_perspective() {
        let game = Game::default()
            .play(Spot { x: 500, y: 500 })
            .unwrap()
            .pass()
            .unwrap();
        let mut rng = StdRng::seed_from_u64(8);
        let mut terminal = Node::new(game.pass().unwrap(), &mut rng);
        // Black passes; White is to move in the finished Black win.
        assert_eq!(terminal.simulate(&SearchConfig::default(), &mut rng), -1.0);
        let config = SearchConfig {
            simulations: 64,
            ..SearchConfig::default()
        };
        let result = search(&game, &config, &mut rng).unwrap();
        let pass = result
            .policy
            .iter()
            .find(|entry| entry.action == Action::PASS)
            .unwrap();
        assert!(
            pass.visits > config.simulations / 2,
            "search should favor the immediate win"
        );
        assert!(search(&terminal.game, &config, &mut rng).is_err());
    }

    #[test]
    fn invalid_search_settings_are_rejected() {
        for config in [
            SearchConfig {
                simulations: 0,
                ..SearchConfig::default()
            },
            SearchConfig {
                cpuct: f64::NAN,
                ..SearchConfig::default()
            },
            SearchConfig {
                widening_exponent: 1.0,
                ..SearchConfig::default()
            },
            SearchConfig {
                proposal_attempts: 0,
                ..SearchConfig::default()
            },
        ] {
            assert!(config.validate().is_err());
        }
    }
}
