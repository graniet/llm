use serde::{Deserialize, Serialize};

/// Usage metadata for a chat response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Number of tokens in the prompt
    #[serde(alias = "input_tokens")]
    pub prompt_tokens: u32,
    /// Number of tokens in the completion
    #[serde(alias = "output_tokens")]
    pub completion_tokens: u32,
    /// Total number of tokens used
    pub total_tokens: u32,
    /// Breakdown of completion tokens, if available
    #[serde(
        skip_serializing_if = "Option::is_none",
        alias = "output_tokens_details"
    )]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    /// Breakdown of prompt tokens, if available
    #[serde(
        skip_serializing_if = "Option::is_none",
        alias = "input_tokens_details"
    )]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

/// Breakdown of completion tokens.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletionTokensDetails {
    /// Tokens used for reasoning (for reasoning models)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<u32>,
    /// Tokens used for audio output
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u32>,
}

/// Breakdown of prompt tokens.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    /// Tokens used for cached content
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
    /// Tokens used for audio input
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_tokens: Option<u32>,
}
