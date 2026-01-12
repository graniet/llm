use llm::secret_store::SecretStore;

use crate::provider::{ProviderId, ProviderSelection};
use crate::runtime::controller::AppController;

use super::super::utils::parse_provider_and_model;
use super::options::TuiOptions;

pub(super) fn apply_launch_options(
    controller: &mut AppController,
    options: &TuiOptions,
) -> anyhow::Result<()> {
    if options.first_run {
        controller.start_onboarding();
        return Ok(());
    }
    if let Some(id) = options.conversation_id {
        if !controller.state.conversations.set_active(id) {
            return Err(anyhow::anyhow!("conversation {id} not found"));
        }
    }
    if options.force_new {
        create_new_conversation(
            controller,
            options.initial_selection.as_ref(),
            options.system_prompt.as_deref(),
        );
        return Ok(());
    }
    ensure_default_conversation(controller, options)?;
    Ok(())
}

fn ensure_default_conversation(
    controller: &mut AppController,
    options: &TuiOptions,
) -> anyhow::Result<()> {
    if controller.state.active_conversation().is_some() {
        return Ok(());
    }
    create_new_conversation(
        controller,
        options.initial_selection.as_ref(),
        options.system_prompt.as_deref(),
    );
    Ok(())
}

fn create_new_conversation(
    controller: &mut AppController,
    selection: Option<&ProviderSelection>,
    system_prompt: Option<&str>,
) {
    let (provider_id, model) = selection
        .map(|sel| (sel.provider_id.clone(), sel.model.clone()))
        .or_else(|| default_provider_selection(controller))
        .unwrap_or((
            ProviderId::new("openai"),
            controller.state.config.default_model.clone(),
        ));
    let system_prompt = system_prompt
        .map(|value| value.to_string())
        .or_else(|| controller.state.config.chat.system_prompt.clone());
    controller
        .state
        .conversations
        .new_conversation(provider_id, model, system_prompt);
    controller.state.scroll.reset();
}

fn default_provider_selection(controller: &AppController) -> Option<(ProviderId, Option<String>)> {
    let default = SecretStore::new()
        .ok()
        .and_then(|store| store.get_default_provider().cloned());
    parse_provider_and_model(default.as_deref()).or_else(|| {
        controller
            .state
            .config
            .default_provider
            .as_deref()
            .and_then(|value| parse_provider_and_model(Some(value)))
    })
}
