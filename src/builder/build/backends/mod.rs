mod anthropic;
mod azure;
mod deepseek;
mod elevenlabs;
mod google;
mod ollama;
mod openai;
mod openai_compatible;
mod phind;
mod xai;

use crate::{
    builder::LLMBackend,
    chat::{Tool, ToolChoice},
    error::LLMError,
    LLMProvider,
};

use crate::builder::state::BuilderState;

pub(super) fn build_backend(
    state: &mut BuilderState,
    backend: LLMBackend,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<ToolChoice>,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    match backend {
        LLMBackend::OpenAI => openai::build_openai(state, tools, tool_choice),
        LLMBackend::Anthropic => anthropic::build_anthropic(state, tools, tool_choice),
        LLMBackend::Ollama => ollama::build_ollama(state, tools),
        LLMBackend::DeepSeek => deepseek::build_deepseek(state),
        LLMBackend::XAI => xai::build_xai(state),
        LLMBackend::Phind => phind::build_phind(state),
        LLMBackend::Google => google::build_google(state, tools),
        LLMBackend::Groq => openai_compatible::build_groq(state, tools, tool_choice),
        LLMBackend::OpenRouter => openai_compatible::build_openrouter(state, tools, tool_choice),
        LLMBackend::Cohere => openai_compatible::build_cohere(state, tools, tool_choice),
        LLMBackend::Mistral => openai_compatible::build_mistral(state, tools, tool_choice),
        LLMBackend::HuggingFace => openai_compatible::build_huggingface(state, tools, tool_choice),
        LLMBackend::AzureOpenAI => azure::build_azure_openai(state, tools, tool_choice),
        LLMBackend::ElevenLabs => elevenlabs::build_elevenlabs(state),
        LLMBackend::AwsBedrock => azure::build_bedrock(state, tools, tool_choice),
    }
}
