mod prompt;
mod runner;
mod selection;
mod stream;
mod tooling;

use crate::args::CliArgs;
use crate::config::AppConfig;
use crate::provider::{ProviderFactory, ProviderRegistry};
use crate::tools::{PtySessionManager, ToolContext, ToolRegistry};

pub async fn run_non_interactive(
    args: &CliArgs,
    config: &AppConfig,
    registry: &ProviderRegistry,
) -> anyhow::Result<()> {
    let prompt = prompt::resolve_prompt(args)?;
    let selection = selection::resolve_provider(args, config)?;

    // Create PTY session manager for shell tools
    let pty_manager = std::sync::Arc::new(PtySessionManager::new());

    // Create diff tracker for rollback support
    let diff_tracker = crate::tools::create_tracker();

    // Create tool registry with PTY support and diff tracking
    let tool_registry =
        ToolRegistry::from_config_with_pty(&config.tools, pty_manager, diff_tracker);

    let working_dir = std::env::current_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_else(|_| ".".to_string());
    let tool_context = ToolContext::new(working_dir)
        .with_allowed_paths(config.tools.allowed_paths.clone())
        .with_timeout(config.tools.timeout_ms);
    let overrides = selection::build_overrides(args, &tool_registry);
    let factory = ProviderFactory::new(config, registry);
    let handle = factory.build(&selection, overrides)?;
    let mut runner = runner::NonInteractiveRunner::new(
        handle,
        tool_registry,
        tool_context,
        config.tools.execution,
    );
    runner.run(prompt).await
}
