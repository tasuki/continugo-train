//! JSON stdin/stdout adapter to shared static Glicko. No games or clocks are
//! persisted here; the caller owns evidence validation, deduplication and storage.

use continugo_rating::{GameResult, Rating};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, io};

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Player {
    id: String,
    rating: f64,
    deviation: f64,
}

impl Player {
    fn state(&self) -> Rating {
        Rating {
            rating: self.rating,
            deviation: self.deviation,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Game {
    black: String,
    white: String,
    /// Black's win points; unresolved games must be omitted, never made draws.
    score: f64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    players: Vec<Player>,
    games: Vec<Game>,
}

#[derive(Serialize)]
struct Response {
    players: Vec<Player>,
}

fn rate(request: Request) -> Result<Response, String> {
    let mut indices = HashMap::new();
    for (index, player) in request.players.iter().enumerate() {
        if player.id.is_empty() || indices.insert(player.id.as_str(), index).is_some() {
            return Err("player IDs must be nonempty and unique".into());
        }
        player.state().validate()?;
    }
    let mut evidence = vec![Vec::new(); request.players.len()];
    for game in request.games {
        let black = *indices
            .get(game.black.as_str())
            .ok_or("unknown black player")?;
        let white = *indices
            .get(game.white.as_str())
            .ok_or("unknown white player")?;
        if black == white {
            return Err("self matches cannot supply rating evidence".into());
        }
        evidence[black].push(GameResult {
            opponent: request.players[white].state(),
            score: game.score,
        });
        evidence[white].push(GameResult {
            opponent: request.players[black].state(),
            score: 1.0 - game.score,
        });
    }
    let players = request
        .players
        .iter()
        .zip(evidence)
        .map(|(player, games)| {
            player.state().update(&games).map(|state| Player {
                id: player.id.clone(),
                rating: state.rating,
                deviation: state.deviation,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Response { players })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let request = serde_json::from_reader(io::stdin().lock())?;
    serde_json::to_writer(io::stdout().lock(), &rate(request)?)?;
    Ok(())
}
