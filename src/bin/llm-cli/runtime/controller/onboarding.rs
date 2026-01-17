use llm::secret_store::SecretStore;

use crate::config::save_config;
use crate::provider::backend_env_key;
use crate::runtime::{AppStatus, OnboardingProvider, OnboardingState, OverlayState};

use super::AppController;

impl AppController {
    pub fn start_onboarding(&mut self) -> bool {
        let providers = build_providers(self);
        let state = OnboardingState::new(
            providers,
            self.state.config.ui.navigation_mode,
            self.state.config.ui.theme.clone(),
        );
        self.state.overlay = OverlayState::Onboarding(state);
        true
    }

    pub fn finish_onboarding_from_overlay(&mut self) -> bool {
        let state = match &self.state.overlay {
            OverlayState::Onboarding(state) => state.clone(),
            _ => return false,
        };
        match complete_onboarding(self, &state) {
            Ok(()) => {
                self.state.overlay = OverlayState::None;
                true
            }
            Err(err) => {
                if let OverlayState::Onboarding(state) = &mut self.state.overlay {
                    state.set_error(err);
                }
                false
            }
        }
    }
}

fn build_providers(controller: &AppController) -> Vec<OnboardingProvider> {
    let mut providers: Vec<_> = controller
        .state
        .provider_registry
        .list()
        .map(|info| OnboardingProvider {
            id: info.id.clone(),
            name: info.display_name.clone(),
            backend: info.backend.clone(),
        })
        .collect();
    providers.sort_by(|a, b| a.name.cmp(&b.name));
    providers
}

fn apply_onboarding_config(
    controller: &mut AppController,
    state: &OnboardingState,
    provider: &OnboardingProvider,
) -> Result<(), String> {
    controller.state.config.default_provider = Some(provider.id.as_str().to_string());
    controller.state.config.ui.navigation_mode = state.mode;
    controller.state.config.ui.theme = state.theme.clone();
    if let Err(err) = save_config(&controller.state.config, &controller.config_paths) {
        controller.set_status(AppStatus::Error(format!("save config: {err}")));
        return Err(format!("save config: {err}"));
    }
    Ok(())
}

fn complete_onboarding(
    controller: &mut AppController,
    state: &OnboardingState,
) -> Result<(), String> {
    let provider = state
        .selected_provider()
        .cloned()
        .ok_or_else(|| "select a provider".to_string())?;
    apply_onboarding_config(controller, state, &provider)?;
    store_default_provider(provider.id.as_str()).map_err(|err| err.to_string())?;
    store_api_key(&provider, state.api_key.trim()).map_err(|err| err.to_string())?;
    controller.state.conversations.new_conversation(
        provider.id.clone(),
        controller.state.config.default_model.clone(),
        controller.state.config.chat.system_prompt.clone(),
    );
    controller.state.scroll.reset();
    controller.record_snapshot();
    Ok(())
}

fn store_default_provider(provider: &str) -> anyhow::Result<()> {
    let mut store = SecretStore::new()?;
    store.set_default_provider(provider)?;
    Ok(())
}

fn store_api_key(provider: &OnboardingProvider, key: &str) -> anyhow::Result<()> {
    let Some(env_key) = backend_env_key(&provider.backend) else {
        return Ok(());
    };
    if key.is_empty() {
        return Ok(());
    }
    let mut store = SecretStore::new()?;
    store.set(env_key, key)?;
    Ok(())
}
