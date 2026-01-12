use serde::{Deserialize, Serialize};

use crate::config::PricingConfig;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub context_window: Option<u32>,
    pub max_output_tokens: Option<u32>,
    pub pricing: Option<PricingConfig>,
    pub supports_tools: Option<bool>,
    pub supports_vision: Option<bool>,
    pub supports_streaming: Option<bool>,
}

impl ModelInfo {
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            context_window: None,
            max_output_tokens: None,
            pricing: None,
            supports_tools: None,
            supports_vision: None,
            supports_streaming: None,
        }
    }
}
