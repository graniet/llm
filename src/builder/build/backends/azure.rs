use crate::{
    chat::{Tool, ToolChoice},
    error::LLMError,
    LLMProvider,
};

use super::super::helpers;
use crate::builder::state::BuilderState;

#[cfg(feature = "azure_openai")]
pub(super) fn build_azure_openai(
    state: &mut BuilderState,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<ToolChoice>,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    let endpoint = state.base_url.take().ok_or_else(|| {
        LLMError::InvalidRequest("No API endpoint provided for Azure OpenAI".into())
    })?;
    let api_key = helpers::require_api_key(state, "Azure OpenAI")?;
    let api_version = state.api_version.take().ok_or_else(|| {
        LLMError::InvalidRequest("No API version provided for Azure OpenAI".to_string())
    })?;
    let deployment = state.deployment_id.take().ok_or_else(|| {
        LLMError::InvalidRequest("No deployment ID provided for Azure OpenAI".into())
    })?;

    let timeout = helpers::timeout_or_default(state);
    let provider = crate::backends::azure_openai::AzureOpenAI::new(
        api_key,
        api_version,
        deployment,
        endpoint,
        state.model.take(),
        state.max_tokens,
        state.temperature,
        timeout,
        state.system.take(),
        state.top_p,
        state.top_k,
        state.embedding_encoding_format.take(),
        state.embedding_dimensions,
        tools,
        tool_choice,
        state.reasoning_effort.take(),
        state.json_schema.take(),
    );

    Ok(Box::new(provider))
}

#[cfg(not(feature = "azure_openai"))]
pub(super) fn build_azure_openai(
    _state: &mut BuilderState,
    _tools: Option<Vec<Tool>>,
    _tool_choice: Option<ToolChoice>,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    Err(LLMError::InvalidRequest(
        "OpenAI feature not enabled".to_string(),
    ))
}

#[cfg(feature = "bedrock")]
pub(super) fn build_bedrock(
    state: &mut BuilderState,
    tools: Option<Vec<Tool>>,
    tool_choice: Option<ToolChoice>,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    let region = state
        .base_url
        .take()
        .ok_or_else(|| LLMError::InvalidRequest("No region provided for AWS Bedrock".into()))?;
    let timeout = helpers::timeout_or_default(state);

    let provider = crate::backends::aws::BedrockBackend::new(
        region,
        state.model.take(),
        state.max_tokens,
        state.temperature,
        timeout,
        state.system.take(),
        state.top_p,
        state.top_k,
        tools,
        tool_choice,
        state.reasoning_effort.take(),
        state.json_schema.take(),
    )?;

    Ok(Box::new(provider))
}

#[cfg(not(feature = "bedrock"))]
pub(super) fn build_bedrock(
    _state: &mut BuilderState,
    _tools: Option<Vec<Tool>>,
    _tool_choice: Option<ToolChoice>,
) -> Result<Box<dyn LLMProvider>, LLMError> {
    Err(LLMError::InvalidRequest(
        "AWS Bedrock feature not enabled".to_string(),
    ))
}
