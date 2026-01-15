use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct ProviderConfig {
    pub backend: Option<String>,
    pub api_key: Option<String>,
    pub api_key_env: Option<String>,
    pub base_url: Option<String>,
    pub model: Option<String>,
    pub models: Vec<ModelConfig>,
    pub enabled: bool,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub timeout_seconds: Option<u64>,
    pub system: Option<String>,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            backend: None,
            api_key: None,
            api_key_env: None,
            base_url: None,
            model: None,
            models: Vec::new(),
            enabled: true,
            temperature: None,
            max_tokens: None,
            timeout_seconds: None,
            system: None,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct ModelConfig {
    pub id: String,
    pub context_window: Option<u32>,
    pub max_output_tokens: Option<u32>,
    pub pricing: Option<PricingConfig>,
    pub supports_tools: Option<bool>,
    pub supports_vision: Option<bool>,
    pub supports_streaming: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PricingConfig {
    pub prompt_per_1k: f32,
    pub completion_per_1k: f32,
}
