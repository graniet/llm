mod prompt;
mod runner;
mod selection;
mod stream;
mod tooling;

use crate::args::CliArgs;
use crate::config::AppConfig;
use crate::provider::{ProviderFactory, ProviderRegistry};
use crate::tools::{ToolContext, ToolRegistry};

pub async fn run_non_interactive(
    args: &CliArgs,
    config: &AppConfig,
    registry: &ProviderRegistry,
) -> anyhow::Result<()> {
    let prompt = prompt::resolve_prompt(args)?;
    let selection = selection::resolve_provider(args, config)?;
    let tool_registry = ToolRegistry::from_config(&config.tools);
    let tool_context = ToolContext {
        allowed_paths: config.tools.allowed_paths.clone(),
        timeout_ms: config.tools.timeout_ms,
        working_dir: std::env::current_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| ".".to_string()),
    };
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
