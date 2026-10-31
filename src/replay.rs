//! SGF-to-position replay for feature extraction. No alternate SGF/rules parser.
//! Targets belong to the PRE-move node. A value label requires natural completion
//! AND a scored RE matching authoritative scoring; truncated games remain unlabeled.

use continugo_rules::{Action, Board, Color, Game, Play, sgf, territory};
use serde::Deserialize;
use std::collections::HashSet;

#[derive(Clone, Debug, Deserialize)]
pub struct PolicyTarget {
    pub action: u32,
    pub visits: u32,
}

pub struct ReplayPosition {
    pub game: Game,
    pub policy: Option<Vec<PolicyTarget>>,
    /// Pre-move side-to-move expected-outcome label, not area margin.
    pub outcome: Option<f32>,
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

pub fn load_position(text: &str, ply: usize) -> Result<ReplayPosition, String> {
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
    if ply > record.plays.len() {
        return Err("ply is beyond the SGF record".into());
    }
    let mut game = Game::new(Board::new(size)?);
    let mut selected = None;
    for (index, &play) in record.plays.iter().enumerate() {
        if index == ply {
            selected = Some(game.clone());
        }
        if play.color() != game.on_move() {
            return Err(format!("wrong player at move {}", index + 1));
        }
        game = match play {
            Play::Place(stone) => game.play(stone.spot),
            Play::Pass(_) => game.pass(),
        }
        .map_err(|e| format!("move {}: {e}", index + 1))?;
    }
    let selected = selected.unwrap_or_else(|| game.clone());
    if selected.is_finished() {
        return Err("selected position is terminal".into());
    }
    let outcome = if let Some(re) = property(&record.root, "RE")? {
        if !game.is_finished() {
            return Err("scored RE on a game that did not finish naturally".into());
        }
        let result = territory::result(territory::areas(game.position().stones()));
        let expected = sgf::format_result(game.board(), result);
        if re != expected {
            return Err(format!(
                "RE {re:?} disagrees with authoritative score {expected:?}"
            ));
        }
        Some(match result {
            territory::Outcome::Draw => 0.0,
            territory::Outcome::Win(color, _) => {
                if color == selected.on_move() {
                    1.0
                } else {
                    -1.0
                }
            }
        })
    } else {
        None
    };
    #[derive(Deserialize)]
    struct Comment {
        policy: Option<Vec<PolicyTarget>>,
    }
    let mut policy = None;
    if let Some(properties) = record.move_properties.get(ply)
        && let Some(comment) = property(properties, "C")?
        && comment.trim_start().starts_with('{')
    {
        policy = serde_json::from_str::<Comment>(comment)
            .map_err(|e| format!("invalid training comment: {e}"))?
            .policy;
    }
    if let Some(targets) = &policy {
        if targets.iter().map(|t| u64::from(t.visits)).sum::<u64>() == 0 {
            return Err("policy target has no positive visit mass".into());
        }
        let prepared = selected.prepare();
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
    Ok(ReplayPosition {
        game: selected,
        policy,
        outcome,
    })
}

pub fn player_index(color: Color) -> u32 {
    u32::from(color == Color::White)
}
