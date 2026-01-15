use llm::builder::LLMBackend;
use llm::secret_store::SecretStore;

use crate::config::{AppConfig, ProviderConfig};
use crate::provider::backend_env_key;
use crate::provider::error::ProviderBuildError;
use crate::provider::resolve::ProviderSelection;

use super::ProviderOverrides;

pub(super) fn resolve_model(
    selection: &ProviderSelection,
    provider_cfg: Option<&ProviderConfig>,
    config: &AppConfig,
    overrides: &ProviderOverrides,
) -> Option<String> {
    overrides
        .model
        .clone()
        .or_else(|| selection.model.clone())
        .or_else(|| provider_cfg.and_then(|cfg| cfg.model.clone()))
        .or_else(|| config.default_model.clone())
}

pub(super) fn resolve_system(
    provider_cfg: Option<&ProviderConfig>,
    config: &AppConfig,
    override_system: Option<&str>,
) -> Option<String> {
    override_system
        .map(|value| value.to_string())
        .or_else(|| provider_cfg.and_then(|cfg| cfg.system.clone()))
        .or_else(|| config.chat.system_prompt.clone())
}

pub(super) fn resolve_temperature(
    provider_cfg: Option<&ProviderConfig>,
    config: &AppConfig,
    override_temp: Option<f32>,
) -> Option<f32> {
    override_temp
        .or_else(|| provider_cfg.and_then(|cfg| cfg.temperature))
        .or(config.chat.temperature)
}

pub(super) fn resolve_max_tokens(
    provider_cfg: Option<&ProviderConfig>,
    config: &AppConfig,
    override_max: Option<u32>,
) -> Option<u32> {
    override_max
        .or_else(|| provider_cfg.and_then(|cfg| cfg.max_tokens))
        .or(config.chat.max_tokens)
}

pub(super) fn resolve_timeout(
    provider_cfg: Option<&ProviderConfig>,
    config: &AppConfig,
    override_timeout: Option<u64>,
) -> Option<u64> {
    override_timeout
        .or_else(|| provider_cfg.and_then(|cfg| cfg.timeout_seconds))
        .or(config.chat.timeout_seconds)
}

pub(super) fn resolve_base_url(
    provider_cfg: Option<&ProviderConfig>,
    override_url: Option<&str>,
) -> Option<String> {
    override_url
        .map(|value| value.to_string())
        .or_else(|| provider_cfg.and_then(|cfg| cfg.base_url.clone()))
}

pub(super) fn resolve_api_key(
    backend: &LLMBackend,
    provider_cfg: Option<&ProviderConfig>,
    override_key: Option<&str>,
    secrets: Option<&SecretStore>,
) -> Result<Option<String>, ProviderBuildError> {
    if let Some(value) = override_key {
        return Ok(Some(value.to_string()));
    }
    if let Some(value) = api_key_from_config_env(provider_cfg) {
        return Ok(Some(value));
    }
    if let Some(value) = api_key_from_backend(backend, secrets) {
        return Ok(Some(value));
    }
    if let Some(value) = provider_cfg.and_then(|cfg| cfg.api_key.clone()) {
        return Ok(Some(value));
    }
    if backend_env_key(backend).is_some() {
        return Err(ProviderBuildError::MissingApiKey(backend_label(backend)));
    }
    Ok(None)
}

fn api_key_from_config_env(provider_cfg: Option<&ProviderConfig>) -> Option<String> {
    let key = provider_cfg.and_then(|cfg| cfg.api_key_env.as_deref())?;
    std::env::var(key).ok()
}

fn api_key_from_backend(backend: &LLMBackend, secrets: Option<&SecretStore>) -> Option<String> {
    let key = backend_env_key(backend)?;
    std::env::var(key)
        .ok()
        .or_else(|| secrets.and_then(|store| store.get(key).cloned()))
}

fn backend_label(backend: &LLMBackend) -> String {
    match backend {
        LLMBackend::OpenAI => "openai",
        LLMBackend::Anthropic => "anthropic",
        LLMBackend::DeepSeek => "deepseek",
        LLMBackend::XAI => "xai",
        LLMBackend::Google => "google",
        LLMBackend::Groq => "groq",
        LLMBackend::AzureOpenAI => "azure-openai",
        LLMBackend::Cohere => "cohere",
        LLMBackend::Mistral => "mistral",
        LLMBackend::OpenRouter => "openrouter",
        LLMBackend::HuggingFace => "huggingface",
        LLMBackend::Ollama => "ollama",
        LLMBackend::Phind => "phind",
        LLMBackend::ElevenLabs => "elevenlabs",
        LLMBackend::AwsBedrock => "aws-bedrock",
    }
    .to_string()
}
