use serde::{Deserialize, Serialize};

use super::{DEFAULT_AUTOSAVE, DEFAULT_AUTO_COMPACT_THRESHOLD, DEFAULT_MAX_CONTEXT_TOKENS};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct ChatConfig {
    pub system_prompt: Option<String>,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub timeout_seconds: Option<u64>,
    pub autosave: bool,
    pub trim_strategy: TrimStrategy,
    pub max_context_tokens: u32,
    pub auto_compact_threshold: f32,
}

impl Default for ChatConfig {
    fn default() -> Self {
        Self {
            system_prompt: None,
            temperature: None,
            max_tokens: None,
            timeout_seconds: None,
            autosave: DEFAULT_AUTOSAVE,
            trim_strategy: TrimStrategy::SlidingWindow,
            max_context_tokens: DEFAULT_MAX_CONTEXT_TOKENS,
            auto_compact_threshold: DEFAULT_AUTO_COMPACT_THRESHOLD,
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TrimStrategy {
    SlidingWindow,
    Summarize,
}
