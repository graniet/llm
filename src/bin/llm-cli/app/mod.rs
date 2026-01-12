mod commands;
mod non_interactive;
mod tui;
mod utils;

use clap::Parser;
use std::io::IsTerminal;

use crate::args::CliArgs;
use crate::config::load_config;
use crate::logging::init_logging;
use crate::provider::ProviderRegistry;

pub async fn run() -> anyhow::Result<()> {
    let args = CliArgs::parse();
    let loaded = load_config(args.config.clone())?;
    init_logging(&loaded.config.logging, &loaded.paths)?;

    if let Some(kind) = args.command_kind() {
        commands::handle_command(kind, &args)?;
        return Ok(());
    }

    let registry = ProviderRegistry::from_config(&loaded.config.providers);
    if args.list_providers {
        commands::list_providers(&registry);
        return Ok(());
    }
    if args.list_models {
        commands::list_models(&args, &loaded.config, &registry).await?;
        return Ok(());
    }

    if args.has_non_interactive_prompt() || !std::io::stdin().is_terminal() {
        non_interactive::run_non_interactive(&args, &loaded.config, &registry).await?;
        return Ok(());
    }

    let mut options = tui::TuiOptions::from_args(&args)?;
    options.first_run = !loaded.config_exists;
    tui::run_tui(loaded.config, loaded.paths, registry, options).await
}
