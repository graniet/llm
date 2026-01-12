use llm::secret_store::SecretStore;

use crate::args::CliArgs;
use crate::config::AppConfig;
use crate::provider::{resolve_selection, ProviderOverrides, ProviderSelection};
use crate::tools::ToolRegistry;

pub(super) fn resolve_provider(
    args: &CliArgs,
    config: &AppConfig,
) -> anyhow::Result<ProviderSelection> {
    let default_provider = SecretStore::new()
        .ok()
        .and_then(|store| store.get_default_provider().cloned());
    resolve_selection(args, config, default_provider).map_err(|err| anyhow::anyhow!(err))
}

pub(super) fn build_overrides(args: &CliArgs, tools: &ToolRegistry) -> ProviderOverrides {
    ProviderOverrides {
        model: args.model.clone(),
        system: args.system.clone(),
        api_key: args.api_key.clone(),
        base_url: args.base_url.clone(),
        temperature: args.temperature,
        max_tokens: args.max_tokens,
        tool_builders: tools.function_builders(),
        ..Default::default()
    }
}
