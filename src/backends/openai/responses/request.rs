use serde::Serialize;

use crate::chat::{ChatMessage, Tool, ToolChoice};
use crate::error::LLMError;
use crate::providers::openai_compatible::{OpenAICompatibleProviderConfig, OpenAIResponseFormat};

use super::super::OpenAITool;
use super::input::{build_input_items, ResponsesInput};

#[derive(Serialize, Debug)]
pub struct OpenAIResponsesRequest {
    pub model: String,
    pub input: ResponsesInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ResponsesReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<ResponsesTextConfig>,
    #[serde(flatten)]
    pub extra_body: serde_json::Map<String, serde_json::Value>,
}

#[derive(Serialize, Debug)]
pub struct ResponsesReasoning {
    pub effort: String,
}

#[derive(Serialize, Debug)]
pub struct ResponsesTextConfig {
    pub format: OpenAIResponseFormat,
}

pub struct ResponsesRequestParams<'a> {
    pub config: &'a OpenAICompatibleProviderConfig,
    pub messages: &'a [ChatMessage],
    pub tools: Option<&'a [Tool]>,
    pub stream: bool,
}

pub struct ResponsesInputRequestParams<'a> {
    pub config: &'a OpenAICompatibleProviderConfig,
    pub input: ResponsesInput,
    pub tools: Option<Vec<OpenAITool>>,
    pub stream: bool,
    pub instructions: Option<String>,
    pub text: Option<ResponsesTextConfig>,
}

pub fn build_responses_request(
    params: ResponsesRequestParams<'_>,
) -> Result<OpenAIResponsesRequest, LLMError> {
    let input = build_request_input(params.messages)?;
    let tools = build_request_tools(params.tools, &params.config.tools);
    let tool_choice = build_request_tool_choice(&tools, &params.config.tool_choice);
    Ok(OpenAIResponsesRequest {
        model: params.config.model.clone(),
        input,
        instructions: params.config.system.clone(),
        max_output_tokens: params.config.max_tokens,
        temperature: params.config.temperature,
        top_p: params.config.top_p,
        top_k: params.config.top_k,
        stream: params.stream,
        tools,
        tool_choice,
        reasoning: build_request_reasoning(params.config.reasoning_effort.as_deref()),
        text: build_request_text(params.config.json_schema.as_ref()),
        extra_body: params.config.extra_body.clone(),
    })
}

pub fn build_responses_request_for_input(
    params: ResponsesInputRequestParams<'_>,
) -> OpenAIResponsesRequest {
    let tool_choice = build_request_tool_choice(&params.tools, &params.config.tool_choice);
    OpenAIResponsesRequest {
        model: params.config.model.clone(),
        input: params.input,
        instructions: params.instructions,
        max_output_tokens: params.config.max_tokens,
        temperature: params.config.temperature,
        top_p: params.config.top_p,
        top_k: params.config.top_k,
        stream: params.stream,
        tools: params.tools,
        tool_choice,
        reasoning: build_request_reasoning(params.config.reasoning_effort.as_deref()),
        text: params.text,
        extra_body: params.config.extra_body.clone(),
    }
}

fn build_request_input(messages: &[ChatMessage]) -> Result<ResponsesInput, LLMError> {
    let items = build_input_items(messages)?;
    Ok(ResponsesInput::Items(items))
}

fn build_request_tools(
    tools: Option<&[Tool]>,
    fallback: &Option<Vec<Tool>>,
) -> Option<Vec<OpenAITool>> {
    let tools = match tools {
        Some(tools) if !tools.is_empty() => Some(tools.to_vec()),
        _ => fallback.clone().filter(|t| !t.is_empty()),
    }
    .unwrap_or_default();

    if tools.is_empty() {
        None
    } else {
        Some(map_function_tools(&tools))
    }
}

fn build_request_tool_choice(
    tools: &Option<Vec<OpenAITool>>,
    tool_choice: &Option<ToolChoice>,
) -> Option<ToolChoice> {
    if tools.is_some() {
        tool_choice.clone()
    } else {
        None
    }
}

fn build_request_reasoning(reasoning_effort: Option<&str>) -> Option<ResponsesReasoning> {
    reasoning_effort.map(|effort| ResponsesReasoning {
        effort: effort.to_string(),
    })
}

fn build_request_text(
    json_schema: Option<&crate::chat::StructuredOutputFormat>,
) -> Option<ResponsesTextConfig> {
    json_schema.cloned().map(|schema| ResponsesTextConfig {
        format: OpenAIResponseFormat::from(schema),
    })
}

fn map_function_tools(tools: &[Tool]) -> Vec<OpenAITool> {
    tools
        .iter()
        .map(|tool| OpenAITool::Function {
            tool_type: tool.tool_type.clone(),
            name: tool.function.name.clone(),
            description: tool.function.description.clone(),
            parameters: tool.function.parameters.clone(),
        })
        .collect()
}

#[cfg(test)]
mod tests;
