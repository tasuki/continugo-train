use crate::{
    action::{Action, Proposals},
    inference::{Evaluation, Evaluator},
};
use continugo_rules::{Color, Game, PreparedGame, territory};
use rand::Rng;
use serde::Serialize;
use std::sync::OnceLock;

pub mod rollout;
use rollout::{RolloutConfig, RolloutStats};

#[derive(Clone, Debug, Serialize)]
pub struct SearchConfig {
    pub simulations: u32,
    pub cpuct: f64,
    pub widening: f64,
    pub widening_exponent: f64,
    pub proposal_attempts: usize,
    /// Neural placement proposal mixture: uniform legal exploration vs policy.
    pub independent_exploration: f64,
    /// Experimental cold-start evaluation; None retains the normal leaf evaluator.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rollout: Option<RolloutConfig>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            simulations: 32,
            cpuct: 1.5,
            widening: 2.0,
            widening_exponent: 0.5,
            proposal_attempts: 128,
            independent_exploration: 0.25,
            rollout: None,
        }
    }
}

impl SearchConfig {
    pub fn validate(&self) -> Result<(), String> {
        if let Some(rollout) = &self.rollout {
            rollout.validate()?;
        }
        if self.simulations == 0
            || !self.cpuct.is_finite()
            || self.cpuct <= 0.0
            || !self.widening.is_finite()
            || self.widening <= 0.0
            || !self.widening_exponent.is_finite()
            || !(0.0..1.0).contains(&self.widening_exponent)
            || self.widening_exponent == 0.0
            || self.proposal_attempts == 0
            || !self.independent_exploration.is_finite()
            || !(0.0..=1.0).contains(&self.independent_exploration)
            || self.independent_exploration == 0.0
        {
            return Err("Search needs positive simulations, cpuct, widening, proposal attempts, 0 < exponent < 1, and 0 < independent_exploration <= 1".into());
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
    pub rollout_stats: RolloutStats,
}

impl SearchResult {
    /// Evaluation play: maximum visits, with random tie breaks.
    pub fn best(&self, rng: &mut impl Rng) -> Action {
        self.policy
            .iter()
            .max_by_key(|visit| (visit.visits, rng.random::<u64>()))
            .expect("nonempty search policy")
            .action
    }

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
    value_visits: u32,
    // Exact evidence about this node's pass action, not a positional heuristic.
    pass_proof: Option<f64>,
    proposals: Proposals,
    prepared: OnceLock<PreparedGame>,
    children: Vec<Edge>,
    neural: Option<SampledPolicy>,
    // Placement importance p(a)/q(a); pass keeps its separate policy stratum.
    importance: f64,
}

impl Node {
    fn new(game: Game, rng: &mut impl Rng) -> Self {
        Self {
            game,
            visits: 0,
            value_sum: 0.0,
            value_visits: 0,
            pass_proof: None,
            proposals: Proposals::new(rng),
            prepared: OnceLock::new(),
            children: Vec::new(),
            neural: None,
            importance: 0.0,
        }
    }

    fn evaluate_neural(
        &mut self,
        evaluator: &mut dyn Evaluator,
        config: &SearchConfig,
        rng: &mut impl Rng,
    ) -> Result<f64, String> {
        let evaluation = evaluator.evaluate(&self.game)?;
        let value = evaluation.value;
        self.neural = Some(SampledPolicy::new(evaluation, config, rng)?);
        Ok(value)
    }

    fn simulate_neural(
        &mut self,
        evaluator: &mut dyn Evaluator,
        config: &SearchConfig,
        rng: &mut impl Rng,
        stats: &mut RolloutStats,
    ) -> Result<Option<f64>, String> {
        if self.game.is_finished() {
            return Ok(self.update_estimate(Some(
                black_outcome(&self.game) * color_sign(self.game.on_move()),
            )));
        }
        self.expand_pass(rng)?;
        if self.pass_proof == Some(1.0) {
            let value = self.children[0]
                .child
                .simulate_neural(evaluator, config, rng, stats)?
                .map(|v| -v);
            return Ok(self.update_estimate(value));
        }
        if self.visits == 0 {
            let value = if let Some(rollout) = &config.rollout {
                rollout::evaluate(&self.game, rollout, config.proposal_attempts, rng, stats)
            } else {
                let value = evaluator.value(&self.game)?;
                if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
                    return Err("invalid neural leaf value".into());
                }
                Some(value)
            };
            return Ok(self.update_estimate(value));
        }
        // Most new leaves are never expanded: do not score a million moves for
        // their value queries. Materialize policy only on a subsequent visit.
        if self.neural.is_none() {
            self.evaluate_neural(evaluator, config, rng)?;
        }
        let policy = self.neural.as_mut().expect("evaluated before expansion");
        let limit = (config.widening * f64::from(self.visits).powf(config.widening_exponent)).ceil()
            as usize;
        if self.children.len() - 1 < limit
            && let Some((action, importance)) = policy.candidates.pop()
        {
            let game = self
                .prepared
                .get_or_init(|| self.game.prepare())
                .play(action.spot().unwrap())?;
            let mut child = Node::new(game, rng);
            child.importance = importance;
            self.children.push(Edge { action, child });
            // A newly proposed action always gets a real evaluation, even if the
            // network undervalues it. This preserves independent exploration.
            let value = self
                .children
                .last_mut()
                .unwrap()
                .child
                .simulate_neural(evaluator, config, rng, stats)?
                .map(|v| -v);
            return Ok(self.update_estimate(value));
        }
        let placement_weight: f64 = self
            .children
            .iter()
            .skip(1)
            .map(|edge| edge.child.importance)
            .sum();
        let exploration = config.cpuct * f64::from(self.visits).sqrt();
        let pass = policy.pass;
        let placement = policy.placement;
        let best = self
            .children
            .iter()
            .enumerate()
            .filter(|(index, _)| {
                *index != 0 || self.pass_proof != Some(-1.0) || self.children.len() == 1
            })
            .map(|(index, edge)| {
                let prior = if index == 0 {
                    pass
                } else if placement_weight > 0.0 {
                    placement * edge.child.importance / placement_weight
                } else {
                    placement / (self.children.len() - 1) as f64
                };
                let score =
                    -edge.child.mean() + exploration * prior / (1.0 + f64::from(edge.child.visits));
                (index, score, rng.random::<u64>())
            })
            .max_by(|a, b| a.1.total_cmp(&b.1).then(a.2.cmp(&b.2)))
            .unwrap()
            .0;
        let value = self.children[best]
            .child
            .simulate_neural(evaluator, config, rng, stats)?
            .map(|v| -v);
        Ok(self.update_estimate(value))
    }

    /// Probe only legal, terminal pass continuations. Never score an unfinished
    /// board or infer a loss from an incomplete set of sampled placement moves.
    fn expand_pass(&mut self, rng: &mut impl Rng) -> Result<(), String> {
        if self.children.is_empty() {
            let next = self.game.pass()?;
            let perspective = color_sign(self.game.on_move());
            self.pass_proof = if next.is_finished() {
                Some(black_outcome(&next) * perspective)
            } else {
                let reply = next.pass()?;
                (black_outcome(&reply) * perspective == -1.0).then_some(-1.0)
            };
            self.children.push(Edge {
                action: Action::PASS,
                child: Node::new(next, rng),
            });
        }
        Ok(())
    }

    fn update_estimate(&mut self, value: Option<f64>) -> Option<f64> {
        self.visits += 1;
        if let Some(value) = value {
            self.value_sum += value;
            self.value_visits += 1;
        }
        value
    }

    fn mean(&self) -> f64 {
        if self.value_visits == 0 {
            0.0 // Explicit lack of evidence, not an observed draw.
        } else {
            self.value_sum / f64::from(self.value_visits)
        }
    }

    fn simulate(
        &mut self,
        config: &SearchConfig,
        rng: &mut impl Rng,
        stats: &mut RolloutStats,
    ) -> Option<f64> {
        if self.game.is_finished() {
            let value = black_outcome(&self.game) * color_sign(self.game.on_move());
            return self.update_estimate(Some(value));
        }
        self.expand_pass(rng).expect("unfinished game");
        if self.pass_proof == Some(1.0) {
            let value = self.children[0]
                .child
                .simulate(config, rng, stats)
                .map(|v| -v);
            return self.update_estimate(value);
        }
        if self.visits == 0 {
            let value = config.rollout.as_ref().map_or(Some(0.0), |rollout| {
                rollout::evaluate(&self.game, rollout, config.proposal_attempts, rng, stats)
            });
            return self.update_estimate(value);
        }
        let placement_limit = (config.widening
            * f64::from(self.visits).powf(config.widening_exponent))
        .ceil() as usize;
        if self.children.len() - 1 < placement_limit
            && let Some((action, game)) = self.proposals.legal(
                self.prepared.get_or_init(|| self.game.prepare()),
                config.proposal_attempts,
            )
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
            .filter(|(index, _)| {
                *index != 0 || self.pass_proof != Some(-1.0) || self.children.len() == 1
            })
            .map(|(index, edge)| {
                let score = -edge.child.mean() + exploration / (1.0 + f64::from(edge.child.visits));
                (index, score, rng.random::<u64>())
            })
            .max_by(|a, b| a.1.total_cmp(&b.1).then(a.2.cmp(&b.2)))
            .unwrap()
            .0;
        let value = self.children[best]
            .child
            .simulate(config, rng, stats)
            .map(|v| -v);
        self.update_estimate(value)
    }
}

/// Importance-corrected sampled-action priors. Pass is enumerated exactly;
/// placements are sampled from q = epsilon*uniform + (1-epsilon)*policy.
/// Rescaling the placement stratum preserves its full-domain mass, avoiding
/// both double-counting policy-guided proposals and making pass dominate a tiny
/// candidate set merely because individual coordinates have tiny probability.
struct SampledPolicy {
    pass: f64,
    placement: f64,
    candidates: Vec<(Action, f64)>,
}

impl SampledPolicy {
    fn new(
        evaluation: Evaluation,
        config: &SearchConfig,
        rng: &mut impl Rng,
    ) -> Result<Self, String> {
        let count = crate::action::COORDINATE_ACTIONS as usize;
        if evaluation.logits.len() != count + 1
            || evaluation.legal.len() != count
            || !evaluation.value.is_finite()
            || !(-1.0..=1.0).contains(&evaluation.value)
            || !evaluation.logits[count].is_finite()
        {
            return Err("invalid neural evaluation".into());
        }
        let mut actions = Vec::new();
        let mut weights = Vec::new();
        let mut maximum = f64::from(evaluation.logits[count]);
        for (index, &legal) in evaluation.legal.iter().enumerate() {
            if legal > 1 {
                return Err("invalid neural legality mask".into());
            }
            if legal == 1 {
                let logit = f64::from(evaluation.logits[index]);
                if !logit.is_finite() {
                    return Err("nonfinite legal neural logit".into());
                }
                maximum = maximum.max(logit);
                actions.push(Action::from_index(index as u32).unwrap());
                weights.push(logit);
            }
        }
        if actions.is_empty() {
            return Ok(Self {
                pass: 1.0,
                placement: 0.0,
                candidates: Vec::new(),
            });
        }
        // Placement-conditional softmax separately avoids underflow if pass is huge.
        let coordinate_max = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        for weight in &mut weights {
            *weight = (*weight - coordinate_max).exp();
        }
        let sum: f64 = weights.iter().sum();
        let placement_mass = (coordinate_max - maximum).exp() * sum;
        let pass_mass = (f64::from(evaluation.logits[count]) - maximum).exp();
        let total = pass_mass + placement_mass;
        let mut cdf = Vec::with_capacity(actions.len());
        let mut importance = Vec::with_capacity(actions.len());
        let mut cumulative = 0.0;
        for weight in weights {
            let conditional = weight / sum;
            let q = config.independent_exploration / actions.len() as f64
                + (1.0 - config.independent_exploration) * conditional;
            cumulative += q;
            cdf.push(cumulative);
            importance.push(conditional / q);
        }
        let limit = (config.widening
            * (f64::from(config.simulations) + 1.0).powf(config.widening_exponent))
        .ceil() as usize;
        let mut candidates = Vec::new();
        // IID draws, merging duplicates with multiplicity. Rejecting duplicates
        // would change q and invalidate the p/q importance correction.
        for _ in 0..limit.min(count) {
            let ticket = rng.random::<f64>() * cumulative;
            let index = cdf
                .partition_point(|&boundary| boundary <= ticket)
                .min(actions.len() - 1);
            let action = actions[index];
            if let Some((_, weight)) = candidates
                .iter_mut()
                .find(|(existing, _)| *existing == action)
            {
                *weight += importance[index];
            } else {
                candidates.push((action, importance[index]));
            }
        }
        candidates.reverse();
        Ok(Self {
            pass: pass_mass / total,
            placement: placement_mass / total,
            candidates,
        })
    }
}

pub fn search_neural(
    game: &Game,
    config: &SearchConfig,
    evaluator: &mut dyn Evaluator,
    rng: &mut impl Rng,
) -> Result<SearchResult, String> {
    config.validate()?;
    if game.is_finished() {
        return Err("Cannot search a finished game".into());
    }
    let mut root = Node::new(game.clone(), rng);
    let value = root.evaluate_neural(evaluator, config, rng)?;
    // In rollout mode the network supplies policy only, never leaf values.
    root.update_estimate(config.rollout.is_none().then_some(value));
    let mut rollout_stats = RolloutStats::default();
    for _ in 0..config.simulations {
        root.simulate_neural(evaluator, config, rng, &mut rollout_stats)?;
    }
    Ok(SearchResult {
        rollout_stats,
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

pub fn color_sign(color: Color) -> f64 {
    match color {
        Color::Black => 1.0,
        Color::White => -1.0,
    }
}

/// The caller must only use this as a target for naturally completed games.
pub fn black_outcome(game: &Game) -> f64 {
    assert!(
        game.is_finished(),
        "authoritative outcome requires natural completion"
    );
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
    root.update_estimate(None); // Structural root visit, not outcome evidence.
    let mut rollout_stats = RolloutStats::default();
    for _ in 0..config.simulations {
        root.simulate(config, rng, &mut rollout_stats);
    }
    Ok(SearchResult {
        rollout_stats,
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
        assert_eq!(
            terminal.simulate(
                &SearchConfig::default(),
                &mut rng,
                &mut RolloutStats::default()
            ),
            Some(-1.0)
        );
        let config = SearchConfig {
            simulations: 32,
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

    struct FixedEvaluator {
        calls: usize,
    }

    impl Evaluator for FixedEvaluator {
        fn evaluate(&mut self, _: &Game) -> Result<Evaluation, String> {
            self.calls += 1;
            let count = crate::action::COORDINATE_ACTIONS as usize;
            let mut logits = vec![f32::NEG_INFINITY; count + 1];
            let mut legal = vec![0; count];
            logits[500_500] = 0.0;
            logits[700_700] = -100.0;
            logits[count] = -10.0;
            legal[500_500] = 1;
            legal[700_700] = 1;
            Ok(Evaluation {
                value: 0.75,
                logits,
                legal,
            })
        }
        fn model_id(&self) -> &str {
            "test"
        }
    }

    #[test]
    fn neural_backup_explores_low_prior_moves_and_does_not_evaluate_terminal_nodes() {
        let config = SearchConfig {
            simulations: 2,
            widening: 100.0,
            independent_exploration: 1.0,
            ..SearchConfig::default()
        };
        let mut evaluator = FixedEvaluator { calls: 0 };
        let mut rng = StdRng::seed_from_u64(7);
        let mut root = Node::new(Game::default(), &mut rng);
        let value = root
            .evaluate_neural(&mut evaluator, &config, &mut rng)
            .unwrap();
        root.update_estimate(Some(value));
        for _ in 0..2 {
            root.simulate_neural(
                &mut evaluator,
                &config,
                &mut rng,
                &mut RolloutStats::default(),
            )
            .unwrap();
        }
        assert_eq!(root.visits, 3);
        assert_eq!(root.value_sum, -0.75); // +root, -child, -child (side-to-move).
        assert_eq!(evaluator.calls, 3);
        assert_eq!(root.children.iter().map(|e| e.child.visits).sum::<u32>(), 2);
        assert!(
            root.children
                .iter()
                .any(|e| e.action.index() == 700_700 && e.child.visits == 1)
        );
        let finished = Game::default()
            .play(Spot { x: 500, y: 500 })
            .unwrap()
            .pass()
            .unwrap()
            .pass()
            .unwrap();
        let mut terminal = Node::new(finished, &mut rng);
        assert_eq!(
            terminal
                .simulate_neural(
                    &mut evaluator,
                    &config,
                    &mut rng,
                    &mut RolloutStats::default()
                )
                .unwrap(),
            Some(-1.0)
        );
        assert_eq!(evaluator.calls, 3);
    }

    #[test]
    fn sampled_priors_preserve_pass_mass_and_merge_iid_duplicate_draws() {
        let count = crate::action::COORDINATE_ACTIONS as usize;
        let evaluation = Evaluation {
            value: 0.0,
            logits: vec![0.0; count + 1],
            legal: vec![1; count],
        };
        let policy = SampledPolicy::new(
            evaluation,
            &SearchConfig::default(),
            &mut StdRng::seed_from_u64(0),
        )
        .unwrap();
        assert!((policy.pass - 1.0 / (count + 1) as f64).abs() < 1e-12);
        assert!((policy.pass + policy.placement - 1.0).abs() < 1e-12);
        assert!(policy.candidates.len() < 20); // Never retain a million-element policy in a node.
        let mut evaluation = FixedEvaluator { calls: 0 }
            .evaluate(&Game::default())
            .unwrap();
        evaluation.logits[700_700] = 0.0;
        evaluation.logits[count] = 2.0_f32.ln(); // pass mass 2 / (1+1+2) = 1/2.
        let config = SearchConfig {
            independent_exploration: 1.0,
            ..SearchConfig::default()
        };
        let policy =
            SampledPolicy::new(evaluation, &config, &mut StdRng::seed_from_u64(0)).unwrap();
        assert!((policy.pass - 0.5).abs() < 1e-7);
        let draws = (config.widening
            * (f64::from(config.simulations) + 1.0).powf(config.widening_exponent))
        .ceil();
        assert_eq!(policy.candidates.len(), 2);
        assert!((policy.candidates.iter().map(|(_, w)| w).sum::<f64>() - draws).abs() < 1e-12);
    }

    #[test]
    #[should_panic(expected = "authoritative outcome requires natural completion")]
    fn unfinished_positions_cannot_be_used_as_authoritative_outcomes() {
        black_outcome(&Game::default());
    }

    struct MisleadingEvaluator {
        pass_logit: f32,
        policy_calls: usize,
        value_calls: usize,
        forbid_value: bool,
    }

    impl Evaluator for MisleadingEvaluator {
        fn evaluate(&mut self, game: &Game) -> Result<Evaluation, String> {
            self.policy_calls += 1;
            let count = crate::action::COORDINATE_ACTIONS as usize;
            let mut logits = vec![f32::NEG_INFINITY; count + 1];
            let mut legal = vec![0; count];
            let prepared = game.prepare();
            for index in [200_200, 500_500, 700_700] {
                if prepared
                    .play(Action::from_index(index).unwrap().spot().unwrap())
                    .is_ok()
                {
                    logits[index as usize] = 0.0;
                    legal[index as usize] = 1;
                }
            }
            logits[count] = self.pass_logit;
            Ok(Evaluation {
                logits,
                legal,
                value: -0.99,
            })
        }
        fn value(&mut self, _: &Game) -> Result<f64, String> {
            self.value_calls += 1;
            if self.forbid_value {
                return Err("cold-start search must not use neural leaf values".into());
            }
            Ok(-0.99)
        }
        fn model_id(&self) -> &str {
            "misleading-test"
        }
    }

    fn misleading(pass_logit: f32) -> MisleadingEvaluator {
        MisleadingEvaluator {
            pass_logit,
            policy_calls: 0,
            value_calls: 0,
            forbid_value: false,
        }
    }

    #[test]
    fn exact_winning_pass_overrides_bad_values_tiny_prior_and_widening() {
        let game = Game::default()
            .play(Spot { x: 500, y: 500 })
            .unwrap()
            .pass()
            .unwrap();
        let mut evaluator = misleading(-80.0);
        let result = search_neural(
            &game,
            &SearchConfig {
                simulations: 4,
                ..SearchConfig::default()
            },
            &mut evaluator,
            &mut StdRng::seed_from_u64(8),
        )
        .unwrap();
        assert_eq!(result.policy.len(), 1);
        assert_eq!(result.policy[0].action, Action::PASS);
        assert_eq!(result.policy[0].visits, 4);
        assert_eq!(evaluator.value_calls, 0);
    }

    #[test]
    fn pass_allowing_an_immediate_winning_reply_is_not_selected_by_huge_prior() {
        let game = Game::default().play(Spot { x: 500, y: 500 }).unwrap();
        let mut evaluator = misleading(80.0);
        let result = search_neural(
            &game,
            &SearchConfig {
                simulations: 32,
                ..SearchConfig::default()
            },
            &mut evaluator,
            &mut StdRng::seed_from_u64(8),
        )
        .unwrap();
        assert_eq!(result.policy.iter().map(|v| v.visits).sum::<u32>(), 32);
        assert_eq!(
            result
                .policy
                .iter()
                .find(|v| v.action == Action::PASS)
                .unwrap()
                .visits,
            0
        );
        assert!(
            result
                .policy
                .iter()
                .any(|v| v.action != Action::PASS && v.visits > 0)
        );
    }

    #[test]
    fn rollout_mode_uses_policy_but_never_neural_leaf_values() {
        let mut evaluator = misleading(0.0);
        evaluator.forbid_value = true;
        let config = SearchConfig {
            simulations: 16,
            widening: 0.5,
            rollout: Some(RolloutConfig {
                playouts: 2,
                max_moves: 2,
                pass_probability: 1.0,
            }),
            ..SearchConfig::default()
        };
        let result = search_neural(
            &Game::default(),
            &config,
            &mut evaluator,
            &mut StdRng::seed_from_u64(8),
        )
        .unwrap();
        assert!(evaluator.policy_calls > 0);
        assert_eq!(evaluator.value_calls, 0);
        assert!(result.rollout_stats.completed > 0);
        assert_eq!(result.policy.iter().map(|v| v.visits).sum::<u32>(), 16);
    }

    #[test]
    fn leading_on_the_board_does_not_prove_a_win_if_opponent_can_reply() {
        let game = Game::default()
            .play(Spot { x: 500, y: 500 })
            .unwrap()
            .pass()
            .unwrap()
            .play(Spot { x: 700, y: 700 })
            .unwrap()
            .pass()
            .unwrap()
            .play(Spot { x: 200, y: 200 })
            .unwrap()
            .play(Spot { x: 800, y: 200 })
            .unwrap();
        let pass = game.pass().unwrap();
        assert!(!pass.is_finished());
        assert_eq!(black_outcome(&pass.pass().unwrap()), 1.0);
        let mut node = Node::new(game, &mut StdRng::seed_from_u64(8));
        node.expand_pass(&mut StdRng::seed_from_u64(8)).unwrap();
        assert_eq!(
            node.pass_proof, None,
            "the opponent is not obliged to cooperate by passing"
        );
    }

    #[test]
    fn unknown_rollouts_do_not_dilute_completed_value_evidence() {
        let mut node = Node::new(Game::default(), &mut StdRng::seed_from_u64(8));
        node.update_estimate(None);
        node.update_estimate(Some(1.0));
        assert_eq!(node.visits, 2);
        assert_eq!(node.value_visits, 1);
        assert_eq!(node.mean(), 1.0);
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
