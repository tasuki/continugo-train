//! Read-only saved-position diagnostic using the shared ContinuGo embedded viewer.
use clap::Parser;
use continugo_train::proposal_view;

#[derive(Parser)]
struct Args {
    #[command(flatten)]
    viewer: proposal_view::Options,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", proposal_view::generate(Args::parse().viewer)?);
    Ok(())
}
