use crate::args::CliArgs;
use crate::conversation::ConversationId;
use crate::provider::{ProviderOverrides, ProviderSelection};

use super::super::utils::parse_provider_and_model;

pub struct TuiOptions {
    pub initial_selection: Option<ProviderSelection>,
    pub conversation_id: Option<ConversationId>,
    pub force_new: bool,
    pub system_prompt: Option<String>,
    pub session_overrides: ProviderOverrides,
    pub first_run: bool,
}

impl TuiOptions {
    pub fn from_args(args: &CliArgs) -> anyhow::Result<Self> {
        let conversation_id = parse_conversation_id(args)?;
        if args.new && conversation_id.is_some() {
            return Err(anyhow::anyhow!(
                "--new cannot be combined with --conversation"
            ));
        }
        Ok(Self {
            initial_selection: selection_from_args(args),
            conversation_id,
            force_new: args.new,
            system_prompt: args.system.clone(),
            session_overrides: build_session_overrides(args),
            first_run: false,
        })
    }
}

fn selection_from_args(args: &CliArgs) -> Option<ProviderSelection> {
    let model_override = args.model.clone();
    selection_from_value(args.provider.as_deref(), model_override.clone())
        .or_else(|| selection_from_command(args, model_override))
}

fn selection_from_value(
    value: Option<&str>,
    model_override: Option<String>,
) -> Option<ProviderSelection> {
    let (provider_id, model) = parse_provider_and_model(value)?;
    Some(ProviderSelection {
        provider_id,
        model: model_override.or(model),
    })
}

fn selection_from_command(
    args: &CliArgs,
    model_override: Option<String>,
) -> Option<ProviderSelection> {
    let command = provider_from_command(args)?;
    let (provider_id, model) = parse_provider_and_model(Some(command))?;
    let model = model_override
        .or(model)
        .or_else(|| args.provider_or_key.clone());
    Some(ProviderSelection { provider_id, model })
}

fn provider_from_command(args: &CliArgs) -> Option<&str> {
    if args.command_kind().is_some() {
        None
    } else {
        args.command.as_deref()
    }
}

fn parse_conversation_id(args: &CliArgs) -> anyhow::Result<Option<ConversationId>> {
    let raw = match args.conversation.as_deref() {
        Some(value) => value,
        None => return Ok(None),
    };
    let uuid = uuid::Uuid::parse_str(raw)
        .map_err(|err| anyhow::anyhow!("invalid conversation id: {err}"))?;
    Ok(Some(ConversationId::from(uuid)))
}

fn build_session_overrides(args: &CliArgs) -> ProviderOverrides {
    ProviderOverrides {
        model: args.model.clone(),
        system: args.system.clone(),
        api_key: args.api_key.clone(),
        base_url: args.base_url.clone(),
        temperature: args.temperature,
        max_tokens: args.max_tokens,
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn refuses_new_with_conversation() {
        let mut args = empty_args();
        args.new = true;
        args.conversation = Some("00000000-0000-0000-0000-000000000000".to_string());
        assert!(TuiOptions::from_args(&args).is_err());
    }

    #[test]
    fn selection_from_command_uses_second_arg_as_model() {
        let mut args = empty_args();
        args.command = Some("openai".to_string());
        args.provider_or_key = Some("gpt-4o-mini".to_string());
        let options = TuiOptions::from_args(&args).unwrap();
        let selection = options.initial_selection.unwrap();
        assert_eq!(selection.provider_id.as_str(), "openai");
        assert_eq!(selection.model.as_deref(), Some("gpt-4o-mini"));
    }
}
