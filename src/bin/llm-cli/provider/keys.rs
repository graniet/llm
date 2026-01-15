use llm::builder::LLMBackend;

pub fn backend_env_key(backend: &LLMBackend) -> Option<&'static str> {
    match backend {
        LLMBackend::OpenAI => Some("OPENAI_API_KEY"),
        LLMBackend::Anthropic => Some("ANTHROPIC_API_KEY"),
        LLMBackend::DeepSeek => Some("DEEPSEEK_API_KEY"),
        LLMBackend::XAI => Some("XAI_API_KEY"),
        LLMBackend::Google => Some("GOOGLE_API_KEY"),
        LLMBackend::Groq => Some("GROQ_API_KEY"),
        LLMBackend::AzureOpenAI => Some("AZURE_OPENAI_API_KEY"),
        LLMBackend::Cohere => Some("COHERE_API_KEY"),
        LLMBackend::Mistral => Some("MISTRAL_API_KEY"),
        LLMBackend::OpenRouter => Some("OPENROUTER_API_KEY"),
        LLMBackend::HuggingFace => Some("HF_TOKEN"),
        LLMBackend::Ollama
        | LLMBackend::Phind
        | LLMBackend::ElevenLabs
        | LLMBackend::AwsBedrock => None,
    }
}
