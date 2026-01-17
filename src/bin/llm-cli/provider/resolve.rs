use crate::args::{CliArgs, CommandKind};
use crate::config::AppConfig;
use crate::provider::id::ProviderId;

#[derive(Debug, Clone)]
pub struct ProviderSelection {
    pub provider_id: ProviderId,
    pub model: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum ResolveError {
    #[error("no provider configured")]
    MissingProvider,
}

pub fn resolve_selection(
    args: &CliArgs,
    config: &AppConfig,
    default_provider: Option<String>,
) -> Result<ProviderSelection, ResolveError> {
    selection_from_args(args)
        .or_else(|| selection_from_default(default_provider.as_deref(), args, config))
        .or_else(|| selection_from_default(config.default_provider.as_deref(), args, config))
        .ok_or(ResolveError::MissingProvider)
}

pub fn parse_provider_string(input: Option<&str>) -> Option<(ProviderId, Option<String>)> {
    let raw = input?;
    let mut parts = raw.splitn(2, ':');
    let provider = parts.next()?.trim();
    if provider.is_empty() {
        return None;
    }
    let model = parts
        .next()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty());
    Some((ProviderId::new(provider), model))
}

fn selection_from_args(args: &CliArgs) -> Option<ProviderSelection> {
    let model_override = args.model.clone();
    selection_from_value(args.provider.as_deref(), model_override.clone())
        .or_else(|| selection_from_command(args, model_override.clone()))
        .or_else(|| selection_from_value(args.provider_or_key.as_deref(), model_override))
}

fn selection_from_command(
    args: &CliArgs,
    model_override: Option<String>,
) -> Option<ProviderSelection> {
    let command = provider_command(args)?;
    selection_from_value(Some(command), model_override)
}

fn provider_command(args: &CliArgs) -> Option<&str> {
    let raw = args.command.as_deref()?;
    if CommandKind::parse(raw).is_some() {
        None
    } else {
        Some(raw)
    }
}

fn selection_from_default(
    raw: Option<&str>,
    args: &CliArgs,
    config: &AppConfig,
) -> Option<ProviderSelection> {
    let model_override = args.model.clone().or(config.default_model.clone());
    selection_from_value(raw, model_override)
}

fn selection_from_value(
    value: Option<&str>,
    model_override: Option<String>,
) -> Option<ProviderSelection> {
    let (provider_id, model) = parse_provider_string(value)?;
    Some(ProviderSelection {
        provider_id,
        model: model_override.or(model),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::args::CliArgs;
    use crate::config::AppConfig;

    fn empty_args() -> CliArgs {
        CliArgs {
            command: None,
            provider_or_key: None,
            prompt_or_value: None,
            provider: None,
            model: None,
            system: None,
            api_key: None,
            base_url: None,
            temperature: None,
            max_tokens: None,
            config: None,
            conversation: None,
            new: false,
            prompt: None,
            list_providers: false,
            list_models: false,
        }
    }

    #[test]
    fn parse_provider_string_handles_model() {
        let (provider, model) = parse_provider_string(Some("openai:gpt-4o")).unwrap();
        assert_eq!(provider.as_str(), "openai");
        assert_eq!(model.as_deref(), Some("gpt-4o"));
    }

    #[test]
    fn resolve_selection_prefers_command_provider() {
        let mut args = empty_args();
        args.command = Some("openai:gpt-4o".to_string());
        let config = AppConfig::default();
        let selection = resolve_selection(&args, &config, None).unwrap();
        assert_eq!(selection.provider_id.as_str(), "openai");
        assert_eq!(selection.model.as_deref(), Some("gpt-4o"));
    }

    #[test]
    fn resolve_selection_uses_config_default() {
        let args = empty_args();
        let config = AppConfig {
            default_provider: Some("openai:gpt-4o-mini".to_string()),
            ..Default::default()
        };
        let selection = resolve_selection(&args, &config, None).unwrap();
        assert_eq!(selection.provider_id.as_str(), "openai");
        assert_eq!(selection.model.as_deref(), Some("gpt-4o-mini"));
    }
}
