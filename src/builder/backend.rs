use std::fmt::Formatter;

use crate::error::LLMError;

/// Supported LLM backend providers.
#[derive(Debug, Clone, PartialEq)]
pub enum LLMBackend {
    OpenAI,
    Anthropic,
    Ollama,
    DeepSeek,
    XAI,
    Phind,
    Google,
    Groq,
    AzureOpenAI,
    ElevenLabs,
    Cohere,
    Mistral,
    OpenRouter,
    HuggingFace,
    AwsBedrock,
}

impl std::str::FromStr for LLMBackend {
    type Err = LLMError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "openai" => Ok(LLMBackend::OpenAI),
            "anthropic" => Ok(LLMBackend::Anthropic),
            "ollama" => Ok(LLMBackend::Ollama),
            "deepseek" => Ok(LLMBackend::DeepSeek),
            "xai" => Ok(LLMBackend::XAI),
            "phind" => Ok(LLMBackend::Phind),
            "google" => Ok(LLMBackend::Google),
            "groq" => Ok(LLMBackend::Groq),
            "azure-openai" => Ok(LLMBackend::AzureOpenAI),
            "elevenlabs" => Ok(LLMBackend::ElevenLabs),
            "cohere" => Ok(LLMBackend::Cohere),
            "mistral" => Ok(LLMBackend::Mistral),
            "openrouter" => Ok(LLMBackend::OpenRouter),
            "huggingface" => Ok(LLMBackend::HuggingFace),
            "aws-bedrock" => Ok(LLMBackend::AwsBedrock),
            _ => Err(LLMError::InvalidRequest(format!(
                "Unknown LLM backend: {s}"
            ))),
        }
    }
}

impl std::fmt::Display for LLMBackend {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let name = match self {
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
        };
        write!(f, "{name}")
    }
}
