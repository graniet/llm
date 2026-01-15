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
