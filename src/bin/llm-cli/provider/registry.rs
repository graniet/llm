use std::collections::BTreeMap;

use llm::builder::LLMBackend;

use super::capabilities::ProviderCapabilities;
use super::id::ProviderId;
use crate::config::ProviderConfig;

#[derive(Debug, Clone)]
pub struct ProviderInfo {
    pub id: ProviderId,
    pub display_name: String,
    pub backend: LLMBackend,
    pub capabilities: ProviderCapabilities,
}

#[derive(Debug)]
pub struct ProviderRegistry {
    providers: BTreeMap<ProviderId, ProviderInfo>,
}

impl ProviderRegistry {
    pub fn from_config(configs: &BTreeMap<String, ProviderConfig>) -> Self {
        let mut providers = builtin_providers();
        for (id, cfg) in configs {
            if cfg.enabled {
                let info = provider_from_config(id, cfg).unwrap_or_else(|| ProviderInfo {
                    id: ProviderId::new(id),
                    display_name: id.to_string(),
                    backend: LLMBackend::OpenAI,
                    capabilities: ProviderCapabilities::FULL,
                });
                providers.insert(info.id.clone(), info);
            }
        }
        Self { providers }
    }

    pub fn list(&self) -> impl Iterator<Item = &ProviderInfo> {
        self.providers.values()
    }

    pub fn get(&self, id: &ProviderId) -> Option<&ProviderInfo> {
        self.providers.get(id)
    }
}

fn provider_from_config(id: &str, cfg: &ProviderConfig) -> Option<ProviderInfo> {
    let backend = cfg.backend.as_ref()?;
    let backend = backend.parse().ok()?;
    let caps = provider_capabilities(&backend);
    Some(ProviderInfo {
        id: ProviderId::new(id),
        display_name: id.to_string(),
        backend,
        capabilities: caps,
    })
}

fn builtin_providers() -> BTreeMap<ProviderId, ProviderInfo> {
    let mut map = BTreeMap::new();
    for info in builtin_provider_list() {
        map.insert(info.id.clone(), info);
    }
    map
}

fn builtin_provider_list() -> Vec<ProviderInfo> {
    vec![
        provider_info("openai", "OpenAI", LLMBackend::OpenAI),
        provider_info("anthropic", "Anthropic", LLMBackend::Anthropic),
        provider_info("google", "Google", LLMBackend::Google),
        provider_info("mistral", "Mistral", LLMBackend::Mistral),
        provider_info("openrouter", "OpenRouter", LLMBackend::OpenRouter),
        provider_info("groq", "Groq", LLMBackend::Groq),
        provider_info("xai", "xAI", LLMBackend::XAI),
        provider_info("deepseek", "DeepSeek", LLMBackend::DeepSeek),
        provider_info("cohere", "Cohere", LLMBackend::Cohere),
        provider_info("azure-openai", "Azure OpenAI", LLMBackend::AzureOpenAI),
        provider_info("ollama", "Ollama", LLMBackend::Ollama),
        provider_info("phind", "Phind", LLMBackend::Phind),
        provider_info("huggingface", "HuggingFace", LLMBackend::HuggingFace),
        provider_info("aws-bedrock", "AWS Bedrock", LLMBackend::AwsBedrock),
        provider_info("elevenlabs", "ElevenLabs", LLMBackend::ElevenLabs),
    ]
}

fn provider_info(id: &str, name: &str, backend: LLMBackend) -> ProviderInfo {
    ProviderInfo {
        id: ProviderId::new(id),
        display_name: name.to_string(),
        backend: backend.clone(),
        capabilities: provider_capabilities(&backend),
    }
}

fn provider_capabilities(backend: &LLMBackend) -> ProviderCapabilities {
    match backend {
        LLMBackend::OpenAI
        | LLMBackend::AzureOpenAI
        | LLMBackend::OpenRouter
        | LLMBackend::Groq
        | LLMBackend::XAI
        | LLMBackend::DeepSeek
        | LLMBackend::Mistral
        | LLMBackend::Cohere
        | LLMBackend::HuggingFace
        | LLMBackend::Anthropic => ProviderCapabilities::FULL,
        LLMBackend::Google | LLMBackend::AwsBedrock => ProviderCapabilities::TOOLS_NO_STREAM,
        LLMBackend::Ollama => ProviderCapabilities::LOCAL_BASIC,
        LLMBackend::Phind => ProviderCapabilities::STREAM_ONLY,
        LLMBackend::ElevenLabs => ProviderCapabilities::NONE,
    }
}
