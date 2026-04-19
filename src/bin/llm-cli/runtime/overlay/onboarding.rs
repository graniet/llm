use llm::builder::LLMBackend;

use crate::config::NavigationMode;
use crate::provider::ProviderId;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum OnboardingStep {
    Welcome,
    Provider,
    ApiKey,
    BaseUrl,
    Preferences,
    Confirm,
}

#[derive(Debug, Clone)]
pub struct OnboardingProvider {
    pub id: ProviderId,
    pub name: String,
    pub backend: LLMBackend,
}

#[derive(Debug, Clone)]
pub struct OnboardingState {
    pub step: OnboardingStep,
    pub providers: Vec<OnboardingProvider>,
    pub selected: usize,
    pub api_key: String,
    pub base_url: String,
    pub mode: NavigationMode,
    pub theme: String,
    pub error: Option<String>,
}

impl OnboardingState {
    pub fn new(providers: Vec<OnboardingProvider>, mode: NavigationMode, theme: String) -> Self {
        Self {
            step: OnboardingStep::Welcome,
            providers,
            selected: 0,
            api_key: String::new(),
            base_url: String::new(),
            mode,
            theme,
            error: None,
        }
    }

    pub fn next_step(&mut self) {
        self.step = match self.step {
            OnboardingStep::Welcome => OnboardingStep::Provider,
            OnboardingStep::Provider => OnboardingStep::ApiKey,
            OnboardingStep::ApiKey => OnboardingStep::BaseUrl,
            OnboardingStep::BaseUrl => OnboardingStep::Preferences,
            OnboardingStep::Preferences => OnboardingStep::Confirm,
            OnboardingStep::Confirm => OnboardingStep::Confirm,
        };
        self.error = None;
    }

    pub fn prev_step(&mut self) {
        self.step = match self.step {
            OnboardingStep::Welcome => OnboardingStep::Welcome,
            OnboardingStep::Provider => OnboardingStep::Welcome,
            OnboardingStep::ApiKey => OnboardingStep::Provider,
            OnboardingStep::BaseUrl => OnboardingStep::ApiKey,
            OnboardingStep::Preferences => OnboardingStep::BaseUrl,
            OnboardingStep::Confirm => OnboardingStep::Preferences,
        };
        self.error = None;
    }

    pub fn select_next(&mut self) {
        if !self.providers.is_empty() {
            self.selected = (self.selected + 1).min(self.providers.len().saturating_sub(1));
        }
    }

    pub fn select_prev(&mut self) {
        if self.selected > 0 {
            self.selected = self.selected.saturating_sub(1);
        }
    }

    pub fn selected_provider(&self) -> Option<&OnboardingProvider> {
        self.providers.get(self.selected)
    }

    pub fn set_error(&mut self, message: impl Into<String>) {
        self.error = Some(message.into());
    }

    pub fn default_base_url(&self) -> Option<&str> {
        self.selected_provider().and_then(|provider| {
            match provider.backend {
                LLMBackend::Anthropic => Some("https://api.anthropic.com/v1"),
                _ => None,
            }
        })
    }
}
