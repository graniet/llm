use serde::Deserialize;

use crate::chat::{ChatResponse, Usage};
use crate::{FunctionCall, ToolCall};

#[derive(Debug, Deserialize)]
pub struct OpenAIResponsesChatResponse {
    pub output: Vec<ResponsesOutputItem>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ResponsesOutputItem {
    #[serde(rename = "message")]
    Message {
        content: Vec<ResponsesOutputContent>,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        id: Option<String>,
        call_id: Option<String>,
        name: String,
        arguments: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ResponsesOutputContent {
    #[serde(rename = "output_text")]
    OutputText { text: String },
    #[serde(other)]
    Other,
}

impl std::fmt::Display for OpenAIResponsesChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut wrote = false;
        if let Some(tool_calls) = self.tool_calls() {
            for tool_call in tool_calls {
                write!(f, "{tool_call}")?;
            }
            wrote = true;
        }
        if let Some(text) = self.text() {
            write!(f, "{text}")?;
            wrote = true;
        }
        if !wrote {
            write!(f, "No response content")?;
        }
        Ok(())
    }
}

impl crate::chat::ChatResponse for OpenAIResponsesChatResponse {
    fn text(&self) -> Option<String> {
        let content = last_message_content(&self.output)?;
        extract_output_text(content)
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        let calls = self
            .output
            .iter()
            .filter_map(response_tool_call)
            .collect::<Vec<_>>();
        if calls.is_empty() {
            None
        } else {
            Some(calls)
        }
    }

    fn usage(&self) -> Option<Usage> {
        self.usage.clone()
    }
}

fn last_message_content(output: &[ResponsesOutputItem]) -> Option<&[ResponsesOutputContent]> {
    output.iter().rev().find_map(|item| match item {
        ResponsesOutputItem::Message { content, .. } => Some(content.as_slice()),
        _ => None,
    })
}

fn extract_output_text(content: &[ResponsesOutputContent]) -> Option<String> {
    let parts = content
        .iter()
        .filter_map(|part| match part {
            ResponsesOutputContent::OutputText { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>();
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(""))
    }
}

fn response_tool_call(item: &ResponsesOutputItem) -> Option<ToolCall> {
    match item {
        ResponsesOutputItem::FunctionCall {
            id,
            call_id,
            name,
            arguments,
        } => {
            let call_id = call_id.clone().or_else(|| id.clone())?;
            Some(ToolCall {
                id: call_id,
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: name.clone(),
                    arguments: arguments.clone(),
                },
            })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests;
