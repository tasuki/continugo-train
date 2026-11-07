//! Validated SGF replay. Targets belong to PRE-move positions. A value label
//! requires natural completion AND RE matching authoritative scoring. ReplayGame
//! retains immutable snapshots so offline loaders replay/score once per game.

use crate::targets::FinalTargets;
use continugo_rules::{Action, Board, COORD_RANGE, Color, Game, Play, sgf, territory};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::HashSet, sync::Arc};

#[derive(Clone, Debug)]
pub struct PolicyTarget {
    pub action: u32,
    pub visits: u32,
}

pub struct ReplayPosition {
    pub game: Game,
    pub policy: Option<Vec<PolicyTarget>>,
    pub outcome: Option<f32>,
    pub final_targets: Option<Arc<FinalTargets>>,
}

pub struct ReplayGame {
    positions: Vec<Game>,
    policies: Vec<Option<Vec<PolicyTarget>>>,
    policy_training: Vec<bool>,
    black_outcome: Option<f32>,
    final_targets: Option<Arc<FinalTargets>>,
    game_id: String,
    root_comment: Option<serde_json::Value>,
    result: Option<String>,
    black_player: Option<String>,
    white_player: Option<String>,
}

/// Metadata, not a second game-record format. Identity hashes SZ + decoded plays,
/// excluding player names, seeds, comments and search targets to prevent split leaks.
#[derive(Debug, Serialize)]
pub struct GameInfo {
    pub game_id: String,
    pub board_size: i32,
    pub diameter: i32,
    pub resolution: usize,
    pub moves: usize,
    pub completed: bool,
    pub labeled: bool,
    pub eligible_plies: Vec<usize>,
    /// Real-policy plies enabled for policy loss, including unlabeled games for audit.
    pub policy_training_plies: Vec<usize>,
    /// Shared-codec metadata for supervised loop artifact validation, not rules.
    pub root_comment: Option<serde_json::Value>,
    pub policy_visit_totals: Vec<Option<u64>>,
    pub result: Option<String>,
    pub black_player: Option<String>,
    pub white_player: Option<String>,
}

fn property<'a>(properties: &'a [sgf::Property], name: &str) -> Result<Option<&'a str>, String> {
    let matching: Vec<_> = properties.iter().filter(|(key, _)| key == name).collect();
    match matching.as_slice() {
        [] => Ok(None),
        [(_, values)] if values.len() == 1 => Ok(Some(&values[0])),
        _ => Err(format!(
            "{name} must occur at most once with exactly one value"
        )),
    }
}

fn policy(
    properties: &[sgf::Property],
    game: &Game,
) -> Result<(Option<Vec<PolicyTarget>>, bool), String> {
    #[derive(Deserialize)]
    struct Comment {
        policy: Option<Vec<(String, u32)>>,
        train_policy: Option<bool>,
    }
    let (targets, train_policy) = match property(properties, "C")? {
        Some(comment) if comment.trim_start().starts_with('{') => {
            let comment = serde_json::from_str::<Comment>(comment)
                .map_err(|e| format!("invalid training comment: {e}"))?;
            if comment.train_policy.is_some() && comment.policy.is_none() {
                return Err("train_policy requires real visit-count policy targets".into());
            }
            (comment.policy, comment.train_policy.unwrap_or(true))
        }
        _ => (None, true),
    };
    let targets = targets
        .map(|pairs| {
            pairs
                .into_iter()
                .map(|(coordinates, visits)| {
                    let action = match sgf::decode_play(&format!("B[{coordinates}]"))? {
                        Play::Pass(_) => Action::PASS.index(),
                        Play::Place(stone) => {
                            let spot = stone.spot;
                            if !(0..COORD_RANGE).contains(&spot.x)
                                || !(0..COORD_RANGE).contains(&spot.y)
                            {
                                return Err("policy coordinates outside action domain".into());
                            }
                            (spot.x + COORD_RANGE * spot.y) as u32
                        }
                    };
                    Ok(PolicyTarget { action, visits })
                })
                .collect::<Result<Vec<_>, String>>()
        })
        .transpose()?;
    if let Some(targets) = &targets {
        if targets.iter().map(|t| u64::from(t.visits)).sum::<u64>() == 0 {
            return Err("policy target has no positive visit mass".into());
        }
        let prepared = game.prepare();
        let mut seen = HashSet::new();
        for target in targets {
            let action =
                Action::from_index(target.action).ok_or("policy action outside action domain")?;
            if !seen.insert(action.index()) {
                return Err("duplicate policy action".into());
            }
            if let Some(spot) = action.spot() {
                prepared
                    .play(spot)
                    .map_err(|e| format!("illegal policy target: {e}"))?;
            }
        }
    }
    let train_policy = train_policy && targets.is_some();
    Ok((targets, train_policy))
}

impl ReplayGame {
    pub fn load(text: &str) -> Result<Self, String> {
        let record = sgf::decode_record(text)?;
        for (name, _) in record
            .root
            .iter()
            .chain(record.move_properties.iter().flatten())
        {
            if ["AB", "AW", "AE", "PL", "HA", "KM", "RU"].contains(&name.as_str()) {
                return Err(format!("unsupported setup/rule property: {name}"));
            }
        }
        let size = property(&record.root, "SZ")?
            .ok_or("SGF is missing SZ")?
            .parse::<i32>()
            .map_err(|e| format!("invalid SZ: {e}"))?;
        let mut game = Game::new(Board::new(size)?);
        let mut positions = Vec::with_capacity(record.plays.len() + 1);
        let mut policies = Vec::with_capacity(record.plays.len());
        let mut policy_training = Vec::with_capacity(record.plays.len());
        let mut identity = Sha256::new();
        identity.update(b"continugo-game\0");
        identity.update(size.to_le_bytes());
        for (index, &play) in record.plays.iter().enumerate() {
            if play.color() != game.on_move() {
                return Err(format!("wrong player at move {}", index + 1));
            }
            if game.is_finished() {
                return Err("move after natural completion".into());
            }
            let (targets, train_policy) = policy(&record.move_properties[index], &game)
                .map_err(|e| format!("move {}: {e}", index + 1))?;
            policies.push(targets);
            policy_training.push(train_policy);
            positions.push(game.clone());
            let action = match play {
                Play::Place(stone) => (stone.spot.x + 1000 * stone.spot.y) as u32,
                Play::Pass(_) => Action::PASS.index(),
            };
            identity.update(action.to_le_bytes());
            game = match play {
                Play::Place(stone) => game.play(stone.spot),
                Play::Pass(_) => game.pass(),
            }
            .map_err(|e| format!("move {}: {e}", index + 1))?;
        }
        let mut final_targets = None;
        let black_outcome = if let Some(re) = property(&record.root, "RE")? {
            if !game.is_finished() {
                return Err("scored RE on a game that did not finish naturally".into());
            }
            let areas = territory::areas(game.position().stones());
            let result = territory::result(areas);
            let expected = sgf::format_result(game.board(), result);
            if re != expected {
                return Err(format!(
                    "RE {re:?} disagrees with authoritative score {expected:?}"
                ));
            }
            final_targets = Some(Arc::new(FinalTargets::new(
                game.position().stones().to_vec(),
                areas,
            )));
            Some(match result {
                territory::Outcome::Draw => 0.0,
                territory::Outcome::Win(Color::Black, _) => 1.0,
                territory::Outcome::Win(Color::White, _) => -1.0,
            })
        } else {
            None
        };
        positions.push(game);
        Ok(Self {
            positions,
            policies,
            policy_training,
            black_outcome,
            final_targets,
            game_id: format!("{:x}", identity.finalize()),
            root_comment: property(&record.root, "C")?
                .and_then(|comment| serde_json::from_str(comment).ok()),
            result: property(&record.root, "RE")?.map(str::to_owned),
            black_player: property(&record.root, "PB")?.map(str::to_owned),
            white_player: property(&record.root, "PW")?.map(str::to_owned),
        })
    }

    pub fn info(&self, samples_per_diameter: usize) -> Result<GameInfo, String> {
        let game = &self.positions[0];
        Ok(GameInfo {
            game_id: self.game_id.clone(),
            board_size: game.board().size(),
            diameter: game.board().diameter(),
            resolution: crate::features::resolution(game.board().diameter(), samples_per_diameter)?,
            moves: self.policies.len(),
            root_comment: self.root_comment.clone(),
            result: self.result.clone(),
            black_player: self.black_player.clone(),
            white_player: self.white_player.clone(),
            policy_visit_totals: self
                .policies
                .iter()
                .map(|policy| {
                    policy
                        .as_ref()
                        .map(|targets| targets.iter().map(|t| u64::from(t.visits)).sum())
                })
                .collect(),
            completed: self.positions.last().unwrap().is_finished(),
            labeled: self.black_outcome.is_some(),
            policy_training_plies: self
                .policy_training
                .iter()
                .enumerate()
                .filter_map(|(ply, &enabled)| enabled.then_some(ply))
                .collect(),
            eligible_plies: self
                .policies
                .iter()
                .enumerate()
                .filter_map(|(ply, policy)| {
                    (self.black_outcome.is_some() && policy.is_some()).then_some(ply)
                })
                .collect(),
        })
    }

    /// Display geometry from authoritative replay, never reconstructed Python rules.
    pub fn viewer_position(&self, ply: usize) -> Result<serde_json::Value, String> {
        let game = self.positions.get(ply).ok_or("ply beyond record")?;
        let play = self
            .positions
            .get(ply + 1)
            .and_then(|next| next.record().last())
            .ok_or("no move at ply")?;
        let action = match play {
            Play::Place(stone) => (stone.spot.x + 1000 * stone.spot.y) as u32,
            Play::Pass(_) => Action::PASS.index(),
        };
        let stones: Vec<_> = game
            .position()
            .stones()
            .iter()
            .map(|stone| (stone.spot.x, stone.spot.y, player_index(stone.color)))
            .collect();
        Ok(serde_json::json!({"stones": stones, "action": action}))
    }

    pub fn position(&self, ply: usize) -> Result<ReplayPosition, String> {
        let game = self
            .positions
            .get(ply)
            .ok_or("ply is beyond the SGF record")?;
        if game.is_finished() {
            return Err("selected position is terminal".into());
        }
        Ok(ReplayPosition {
            game: game.clone(),
            policy: self.policies.get(ply).cloned().flatten(),
            final_targets: self.final_targets.clone(),
            outcome: self.black_outcome.map(|value| {
                if game.on_move() == Color::Black {
                    value
                } else {
                    -value
                }
            }),
        })
    }
}

pub fn load_position(text: &str, ply: usize) -> Result<ReplayPosition, String> {
    ReplayGame::load(text)?.position(ply)
}

pub fn player_index(color: Color) -> u32 {
    u32::from(color == Color::White)
}
