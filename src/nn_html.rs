//! Opt-in post-game checkpoint microscope. Never changes search RNG or labels.
//!
//! Full viewer semantics and runtime requirements: docs/current/nn-viewer.md.
use clap::Args;
use std::{
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Args, Debug, Default)]
pub struct NnHtmlArgs {
    /// Write a self-contained .nn.html microscope beside each SGF (slow diagnostic replay).
    #[arg(long, requires = "nn_checkpoint")]
    pub nn_html: bool,
    /// Matching PyTorch checkpoint (self-play model or arena candidate); needed for ownership.
    #[arg(long, requires = "nn_html")]
    pub nn_checkpoint: Option<PathBuf>,
    /// Arena incumbent's matching PyTorch checkpoint (not needed versus random).
    #[arg(long, requires = "nn_html")]
    pub nn_opponent_checkpoint: Option<PathBuf>,
}

impl NnHtmlArgs {
    pub fn validate(&self, arena: bool, neural_opponent: bool) -> Result<(), String> {
        if !self.nn_html {
            return Ok(());
        }
        if arena && neural_opponent && self.nn_opponent_checkpoint.is_none() {
            return Err("--nn-html with an incumbent requires --nn-opponent-checkpoint".into());
        }
        if self.nn_opponent_checkpoint.is_some() && (!arena || !neural_opponent) {
            return Err("--nn-opponent-checkpoint requires an arena neural incumbent".into());
        }
        for path in self
            .nn_checkpoint
            .iter()
            .chain(self.nn_opponent_checkpoint.iter())
        {
            if !path.is_file() {
                return Err(format!("NN checkpoint missing: {}", path.display()));
            }
        }
        Ok(())
    }

    /// Arena candidates alternate colors. On random turns the candidate network
    /// is shown explicitly as a counterfactual diagnostic, not a random-bot NN.
    pub fn render(&self, sgf: &Path, candidate_black: bool) -> Result<(), String> {
        if !self.nn_html {
            return Ok(());
        }
        let candidate = self.nn_checkpoint.as_ref().ok_or("missing NN checkpoint")?;
        let opponent = self.nn_opponent_checkpoint.as_ref().unwrap_or(candidate);
        let (black, white) = if candidate_black {
            (candidate, opponent)
        } else {
            (opponent, candidate)
        };
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let python = std::env::var_os("CONTINUGO_PYTHON")
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join(".venv/bin/python"));
        let features = std::env::var_os("CONTINUGO_FEATURE_BIN")
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join("target/release/features"));
        let status = Command::new(python)
            .args(["-m", "continugo_train.nn_view", "--sgf"])
            .arg(sgf)
            .arg("--checkpoint")
            .arg(black)
            .arg("--white-checkpoint")
            .arg(white)
            .arg("--feature-binary")
            .arg(features)
            .status()
            .map_err(|e| format!("NN HTML replay: {e}; run mise run setup"))?;
        if !status.success() {
            return Err(format!(
                "NN HTML replay failed for {}; SGF is preserved",
                sgf.display()
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        viewer: NnHtmlArgs,
    }

    #[test]
    fn flags_require_explicit_opt_in_and_matching_checkpoints() {
        assert!(!Cli::parse_from(["test"]).viewer.nn_html);
        assert!(Cli::try_parse_from(["test", "--nn-html"]).is_err());
        assert!(Cli::try_parse_from(["test", "--nn-checkpoint", "x"]).is_err());
        let mut args =
            Cli::parse_from(["test", "--nn-html", "--nn-checkpoint", "Cargo.toml"]).viewer;
        assert!(args.validate(false, false).is_ok());
        assert!(args.validate(true, true).unwrap_err().contains("opponent"));
        args.nn_opponent_checkpoint = Some("Cargo.toml".into());
        assert!(args.validate(true, true).is_ok());
        assert!(args.validate(true, false).is_err());
        assert!(args.validate(false, false).is_err());
    }
}
