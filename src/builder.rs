#[path = "builder/backend.rs"]
mod backend;

#[path = "builder/llm_builder.rs"]
mod llm_builder;

#[path = "builder/validation.rs"]
mod validation;

#[path = "builder/tools.rs"]
mod tools;

#[path = "builder/state.rs"]
mod state;

#[path = "builder/build/mod.rs"]
mod build;

#[path = "builder/memory.rs"]
mod memory;

#[path = "builder/resilience.rs"]
mod resilience;

#[path = "builder/search.rs"]
mod search;

#[path = "builder/azure.rs"]
mod azure;

#[path = "builder/embedding.rs"]
mod embedding;

#[path = "builder/voice.rs"]
mod voice;

use serde::Serialize;
use serde_json::Value;

/// Content object for structured system prompts with optional cache control.
///
/// This allows fine-grained control over system prompt components, particularly
/// useful for providers like Anthropic that support prompt caching.
#[derive(Debug, Clone, Serialize)]
pub struct SystemContent {
    pub text: String,
    #[serde(rename = "type")]
    pub content_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<Value>,
}

impl SystemContent {
    /// Creates a new text content segment for system prompts.
    pub fn text(text: String) -> Self {
        Self {
            text,
            content_type: "text".to_string(),
            cache_control: None,
        }
    }

    /// Creates a new text content segment with cache control.
    pub fn text_with_cache(text: String, cache_control: Value) -> Self {
        Self {
            text,
            content_type: "text".to_string(),
            cache_control: Some(cache_control),
        }
    }
}

/// System prompt configuration supporting both simple strings and structured message formats.
///
/// This enum allows system prompts to be specified either as a simple string or a vector
/// of content objects with fine-grained control, particularly useful for caching parts
/// of the system prompt with certain providers.
#[derive(Debug, Clone)]
pub enum SystemPrompt {
    String(String),
    Messages(Vec<SystemContent>),
}

impl SystemPrompt {
    /// Converts the system prompt to a string representation.
    ///
    /// For `SystemPrompt::String`, returns the string directly.
    /// For `SystemPrompt::Messages`, concatenates all message text with newlines.
    pub fn to_string_representation(self) -> String {
        match self {
            SystemPrompt::String(s) => s,
            SystemPrompt::Messages(messages) => messages
                .into_iter()
                .map(|m| m.text)
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

impl From<SystemPrompt> for String {
    fn from(prompt: SystemPrompt) -> String {
        prompt.to_string_representation()
    }
}

pub trait IntoSystemMessage {
    fn into_system_message(self) -> SystemPrompt;
}

impl IntoSystemMessage for String {
    fn into_system_message(self) -> SystemPrompt {
        SystemPrompt::String(self)
    }
}

impl IntoSystemMessage for &str {
    fn into_system_message(self) -> SystemPrompt {
        SystemPrompt::String(self.to_string())
    }
}

impl IntoSystemMessage for Vec<String> {
    fn into_system_message(self) -> SystemPrompt {
        if self.len() == 1 {
            SystemPrompt::String(self.into_iter().next().unwrap())
        } else {
            SystemPrompt::Messages(self.into_iter().map(SystemContent::text).collect())
        }
    }
}

impl IntoSystemMessage for Vec<&str> {
    fn into_system_message(self) -> SystemPrompt {
        if self.len() == 1 {
            SystemPrompt::String(self[0].to_string())
        } else {
            SystemPrompt::Messages(
                self.into_iter()
                    .map(|s| SystemContent::text(s.to_string()))
                    .collect(),
            )
        }
    }
}

impl IntoSystemMessage for Vec<SystemContent> {
    fn into_system_message(self) -> SystemPrompt {
        SystemPrompt::Messages(self)
    }
}

impl IntoSystemMessage for SystemPrompt {
    fn into_system_message(self) -> SystemPrompt {
        self
    }
}

pub use backend::LLMBackend;
pub use llm_builder::LLMBuilder;
pub use tools::{FunctionBuilder, ParamBuilder};
pub use validation::ValidatorFn;
