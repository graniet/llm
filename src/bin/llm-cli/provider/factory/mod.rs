mod build;
mod resolve;

use std::sync::Arc;

use llm::builder::FunctionBuilder;
use llm::secret_store::SecretStore;
use llm::LLMProvider;

use crate::config::{AppConfig, ProviderConfig};
use crate::provider::capabilities::ProviderCapabilities;
use crate::provider::error::ProviderBuildError;
use crate::provider::id::ProviderId;
use crate::provider::registry::{ProviderInfo, ProviderRegistry};

use super::resolve::ProviderSelection;

#[derive(Default)]
pub struct ProviderOverrides {
    pub model: Option<String>,
    pub system: Option<String>,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub timeout_seconds: Option<u64>,
    pub tool_builders: Vec<FunctionBuilder>,
}

impl ProviderOverrides {
    pub fn with_tools(&self, tool_builders: Vec<FunctionBuilder>) -> Self {
        Self {
            model: self.model.clone(),
            system: self.system.clone(),
            api_key: self.api_key.clone(),
            base_url: self.base_url.clone(),
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            timeout_seconds: self.timeout_seconds,
            tool_builders,
        }
    }

    fn split_tools(self) -> (Self, Vec<FunctionBuilder>) {
        let ProviderOverrides {
            model,
            system,
            api_key,
            base_url,
            temperature,
            max_tokens,
            timeout_seconds,
            tool_builders,
        } = self;
        let overrides = ProviderOverrides {
            model,
            system,
            api_key,
            base_url,
            temperature,
            max_tokens,
            timeout_seconds,
            tool_builders: Vec::new(),
        };
        (overrides, tool_builders)
    }
}

#[derive(Clone)]
pub struct ProviderHandle {
    pub id: ProviderId,
    pub capabilities: ProviderCapabilities,
    pub provider: Arc<dyn LLMProvider>,
}

pub struct ProviderFactory<'a> {
    config: &'a AppConfig,
    registry: &'a ProviderRegistry,
    secrets: Option<SecretStore>,
}

impl<'a> ProviderFactory<'a> {
    pub fn new(config: &'a AppConfig, registry: &'a ProviderRegistry) -> Self {
        Self {
            config,
            registry,
            secrets: SecretStore::new().ok(),
        }
    }

    pub fn build(
        &self,
        selection: &ProviderSelection,
        overrides: ProviderOverrides,
    ) -> Result<ProviderHandle, ProviderBuildError> {
        let info = self.provider_info(selection)?;
        let (overrides, tool_builders) = overrides.split_tools();
        let provider_cfg = self.config.providers.get(info.id.as_str());
        let resolved = self.resolve_config(selection, provider_cfg, &overrides, info)?;
        let provider = build::build_provider(&info.backend, &resolved, tool_builders)?;
        Ok(self.build_handle(info, provider))
    }

    fn provider_info(
        &self,
        selection: &ProviderSelection,
    ) -> Result<&ProviderInfo, ProviderBuildError> {
        self.registry
            .get(&selection.provider_id)
            .ok_or_else(|| ProviderBuildError::UnknownProvider(selection.provider_id.to_string()))
    }

    fn resolve_config(
        &self,
        selection: &ProviderSelection,
        provider_cfg: Option<&ProviderConfig>,
        overrides: &ProviderOverrides,
        info: &ProviderInfo,
    ) -> Result<ResolvedConfig, ProviderBuildError> {
        let model = resolve::resolve_model(selection, provider_cfg, self.config, overrides);
        let api_key = resolve::resolve_api_key(
            &info.backend,
            provider_cfg,
            overrides.api_key.as_deref(),
            self.secrets.as_ref(),
        )?;
        Ok(ResolvedConfig {
            model,
            system: resolve::resolve_system(provider_cfg, self.config, overrides.system.as_deref()),
            api_key,
            base_url: resolve::resolve_base_url(provider_cfg, overrides.base_url.as_deref()),
            temperature: resolve::resolve_temperature(
                provider_cfg,
                self.config,
                overrides.temperature,
            ),
            max_tokens: resolve::resolve_max_tokens(
                provider_cfg,
                self.config,
                overrides.max_tokens,
            ),
            timeout_seconds: resolve::resolve_timeout(
                provider_cfg,
                self.config,
                overrides.timeout_seconds,
            ),
        })
    }

    fn build_handle(&self, info: &ProviderInfo, provider: Box<dyn LLMProvider>) -> ProviderHandle {
        ProviderHandle {
            id: info.id.clone(),
            capabilities: info.capabilities,
            provider: Arc::from(provider),
        }
    }
}

struct ResolvedConfig {
    model: Option<String>,
    system: Option<String>,
    api_key: Option<String>,
    base_url: Option<String>,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    timeout_seconds: Option<u64>,
}
