mod cohere;
mod groq;
mod huggingface;
mod mistral;
mod openrouter;

use crate::{
    builder::state::BuilderState,
    chat::{Tool, ToolChoice},
    error::LLMError,
    LLMProvider,
};

pub(super) fn build_groq(
    state: &mut BuilderState,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<ToolChoice>,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    groq::build_groq(state, tools, tool_choice)
}

pub(super) fn build_openrouter(
    state: &mut BuilderState,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<ToolChoice>,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    openrouter::build_openrouter(state, tools, tool_choice)
}

pub(super) fn build_huggingface(
    state: &mut BuilderState,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<ToolChoice>,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    huggingface::build_huggingface(state, tools, tool_choice)
}

pub(super) fn build_mistral(
    state: &mut BuilderState,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<ToolChoice>,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    mistral::build_mistral(state, tools, tool_choice)
}

pub(super) fn build_cohere(
    state: &mut BuilderState,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<ToolChoice>,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    cohere::build_cohere(state, tools, tool_choice)
}
