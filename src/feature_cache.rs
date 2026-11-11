//! Post-generation, lossless training-feature sidecars. Python owns the container;
//! Rust still owns all parsing, rules and feature extraction via `features`.
use clap::Args;
use std::{
    path::{Path, PathBuf},
    process::Command,
};

#[derive(Args, Debug)]
pub struct FeatureCacheArgs {
    /// Skip automatic training-feature sidecars (SGFs are still saved).
    #[arg(long)]
    pub no_feature_cache: bool,
    /// Parallel post-generation cache builders; independent of inference threads.
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u32).range(1..))]
    pub feature_cache_workers: u32,
    /// Raster samples per stone diameter in training-feature sidecars.
    #[arg(long, default_value_t = 4, value_parser = clap::value_parser!(u32).range(1..))]
    pub feature_cache_density: u32,
}

impl FeatureCacheArgs {
    fn command(&self, directory: &Path) -> Command {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let python = std::env::var_os("CONTINUGO_PYTHON")
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join(".venv/bin/python"));
        let features = std::env::var_os("CONTINUGO_FEATURE_BIN")
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join("target/release/features"));
        let mut command = Command::new(python);
        command
            .args(["-m", "continugo_train.feature_cache", "--data"])
            .arg(directory)
            .arg("--skip-ineligible")
            .arg("--feature-binary")
            .arg(features)
            .arg("--workers")
            .arg(self.feature_cache_workers.to_string())
            .arg("--density")
            .arg(self.feature_cache_density.to_string());
        command
    }

    pub fn build(&self, directory: &Path) -> Result<(), String> {
        if self.no_feature_cache {
            return Ok(());
        }
        eprintln!(
            "Building training-feature sidecars in {}",
            directory.display()
        );
        // On Unix replace the completed generator: its PID/process group and
        // campaign lock remain owned by the cache job. An interrupted supervisor
        // must not mistake an exited generator for a finished cache subprocess.
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            let error = self.command(directory).exec();
            Err(format!(
                "Feature cache build: {error}; SGFs are preserved; run mise run setup \
                 or use --no-feature-cache"
            ))
        }
        #[cfg(not(unix))]
        {
            let status = self
                .command(directory)
                .status()
                .map_err(|error| format!("Feature cache build: {error}; SGFs are preserved"))?;
            if !status.success() {
                return Err(format!(
                    "Feature cache build failed in {}; SGFs are preserved",
                    directory.display()
                ));
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        cache: FeatureCacheArgs,
    }

    #[test]
    fn building_is_default_with_an_explicit_opt_out() {
        let args = Cli::parse_from(["test"]).cache;
        assert!(!args.no_feature_cache);
        assert_eq!(args.feature_cache_workers, 4);
        assert_eq!(args.feature_cache_density, 4);
        let command = args.command(Path::new("some games"));
        let argv: Vec<_> = command
            .get_args()
            .map(|arg| arg.to_string_lossy())
            .collect();
        assert!(argv.windows(2).any(|pair| pair == ["--data", "some games"]));
        assert!(argv.iter().any(|arg| arg == "--skip-ineligible"));
        let disabled = Cli::parse_from(["test", "--no-feature-cache"]).cache;
        // Must return without launching Python or accessing a directory.
        assert!(disabled.build(Path::new("does-not-exist")).is_ok());
        assert!(Cli::try_parse_from(["test", "--feature-cache-workers", "0"]).is_err());
        assert!(Cli::try_parse_from(["test", "--feature-cache-density", "0"]).is_err());
    }
}
