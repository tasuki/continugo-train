//! Opt-in wall-clock instrumentation. Stage times are disjoint; the remaining
//! move time includes proposal iteration, bookkeeping, and instrumentation overhead.

use serde::Serialize;
use std::time::Instant;

#[derive(Debug, Default, Serialize)]
pub struct Stage {
    pub calls: u64,
    pub seconds: f64,
}

impl Stage {
    pub fn record(&mut self, started: Instant) -> f64 {
        let seconds = started.elapsed().as_secs_f64();
        self.calls += 1;
        self.seconds += seconds;
        seconds
    }
}

#[derive(Debug, Default, Serialize)]
pub struct MoveProfile {
    pub ply: usize,
    pub stones_before: usize,
    pub geometric_candidates: usize,
    pub passed: bool,
    pub seconds: f64,
    pub proposals: u64,
    pub legal: u64,
    pub rejected_eyes: u64,
    /// Whole Action::apply, including geometry, groups/liberties/captures,
    /// cloning, and construction of illegal-move errors.
    pub rules: Stage,
    /// Subtotals of rules.seconds, classified by the returned move result.
    pub legal_rules_seconds: f64,
    pub illegal_rules_seconds: f64,
    pub capture_check: Stage,
    pub eyes: Stage,
    pub availability_update: Stage,
}

#[derive(Debug, Default, Serialize)]
pub struct GameProfile {
    // These complete after SGF serialization and are reported only in stderr.
    #[serde(skip_serializing)]
    pub seconds: f64,
    pub selfplay_seconds: f64,
    pub availability_init_seconds: f64,
    pub scoring_seconds: f64,
    #[serde(skip_serializing)]
    pub sgf_seconds: f64,
    /// Serialized alongside each move, not duplicated in the root comment.
    #[serde(skip_serializing)]
    pub moves: Vec<MoveProfile>,
}

impl GameProfile {
    pub fn summary(&self) -> String {
        let rules: f64 = self.moves.iter().map(|m| m.rules.seconds).sum();
        let eyes: f64 = self.moves.iter().map(|m| m.eyes.seconds).sum();
        let capture_check: f64 = self.moves.iter().map(|m| m.capture_check.seconds).sum();
        let updates: f64 = self
            .moves
            .iter()
            .map(|m| m.availability_update.seconds)
            .sum();
        let other = (self.selfplay_seconds
            - rules
            - eyes
            - capture_check
            - updates
            - self.availability_init_seconds)
            .max(0.0);
        let proposals: u64 = self.moves.iter().map(|m| m.proposals).sum();
        let legal: u64 = self.moves.iter().map(|m| m.legal).sum();
        let eye_calls: u64 = self.moves.iter().map(|m| m.eyes.calls).sum();
        let rejected: u64 = self.moves.iter().map(|m| m.rejected_eyes).sum();
        let legal_rules: f64 = self.moves.iter().map(|m| m.legal_rules_seconds).sum();
        let illegal_rules: f64 = self.moves.iter().map(|m| m.illegal_rules_seconds).sum();
        let mut lines = vec![format!(
            "profile: {proposals} proposals, {legal} legal, {eye_calls} eye checks, {rejected} eye rejections"
        )];
        for (name, seconds) in [
            ("rules / move application", rules),
            ("small-eye detection", eyes),
            ("capture bookkeeping", capture_check),
            (
                "availability initialization",
                self.availability_init_seconds,
            ),
            ("availability updates", updates),
            ("proposal / other / profiler overhead", other),
            ("final scoring", self.scoring_seconds),
            ("SGF export", self.sgf_seconds),
        ] {
            let percentage = if self.seconds > 0.0 {
                100.0 * seconds / self.seconds
            } else {
                0.0
            };
            lines.push(format!("  {name}: {seconds:.6}s ({percentage:.1}%)"));
        }
        lines.push(format!(
            "  rules subtotal: legal={legal_rules:.6}s, illegal={illegal_rules:.6}s"
        ));
        let mut slowest: Vec<_> = self.moves.iter().collect();
        slowest.sort_by(|a, b| b.seconds.total_cmp(&a.seconds));
        for m in slowest.into_iter().take(5) {
            lines.push(format!(
                "  slow move {}: {:.3}s, {} stones, {}/{} free candidates tried, rules={:.3}s, eyes={:.3}s, pass={}",
                m.ply, m.seconds, m.stones_before, m.proposals, m.geometric_candidates,
                m.rules.seconds, m.eyes.seconds, m.passed
            ));
        }
        lines.join("\n")
    }
}
