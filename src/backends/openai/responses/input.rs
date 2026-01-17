use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde::Serialize;

use crate::chat::{ChatMessage, ChatRole, ImageMime, MessageType};
use crate::error::LLMError;
use crate::ToolCall;

#[derive(Serialize, Debug)]
#[serde(untagged)]
pub enum ResponsesInput {
    Text(String),
    Items(Vec<ResponsesInputItem>),
}

#[derive(Serialize, Debug)]
#[serde(untagged)]
pub enum ResponsesInputItem {
    Message(ResponsesInputMessage),
    FunctionCall(ResponsesFunctionCallItem),
    FunctionCallOutput(ResponsesFunctionCallOutputItem),
}

#[derive(Serialize, Debug)]
pub struct ResponsesInputMessage {
    pub role: String,
    pub content: Vec<ResponsesInputContent>,
}

#[derive(Serialize, Debug)]
#[serde(tag = "type")]
pub enum ResponsesInputContent {
    #[serde(rename = "input_text")]
    Text { text: String },
    #[serde(rename = "output_text")]
    OutputText { text: String },
    #[serde(rename = "input_image")]
    Image { image_url: String },
}

#[derive(Serialize, Debug)]
pub struct ResponsesFunctionCallItem {
    #[serde(rename = "type")]
    pub item_type: ResponsesFunctionCallItemType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub call_id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesFunctionCallItemType {
    FunctionCall,
}

#[derive(Serialize, Debug)]
pub struct ResponsesFunctionCallOutputItem {
    #[serde(rename = "type")]
    pub item_type: ResponsesFunctionCallOutputItemType,
    pub call_id: String,
    pub output: String,
}

#[derive(Serialize, Debug)]
#[serde(rename_all = "snake_case")]
pub enum ResponsesFunctionCallOutputItemType {
    FunctionCallOutput,
}

pub fn build_input_items(messages: &[ChatMessage]) -> Result<Vec<ResponsesInputItem>, LLMError> {
    let mut items = Vec::new();
    for message in messages {
        items.append(&mut map_message_items(message)?);
    }
    Ok(items)
}

fn map_message_items(message: &ChatMessage) -> Result<Vec<ResponsesInputItem>, LLMError> {
    match &message.message_type {
        MessageType::ToolUse(calls) => Ok(map_tool_calls(calls)),
        MessageType::ToolResult(results) => Ok(map_tool_results(results)),
        _ => Ok(vec![ResponsesInputItem::Message(map_message(message)?)]),
    }
}

fn map_message(message: &ChatMessage) -> Result<ResponsesInputMessage, LLMError> {
    let content = map_message_content(message)?;
    Ok(ResponsesInputMessage {
        role: map_role(&message.role).to_string(),
        content,
    })
}

fn map_role(role: &ChatRole) -> &'static str {
    match role {
        ChatRole::User => "user",
        ChatRole::Assistant => "assistant",
    }
}

fn map_message_content(message: &ChatMessage) -> Result<Vec<ResponsesInputContent>, LLMError> {
    match &message.message_type {
        MessageType::Text => Ok(vec![text_content_for_role(
            &message.role,
            message.content.as_str(),
        )]),
        MessageType::ImageURL(url) => {
            map_image_message(&message.role, message.content.as_str(), url)
        }
        MessageType::Image((mime, bytes)) => map_image_message(
            &message.role,
            message.content.as_str(),
            &image_data_url(mime, bytes),
        ),
        MessageType::Pdf(_) => Err(LLMError::InvalidRequest(
            "OpenAI responses PDF input requires file upload".to_string(),
        )),
        MessageType::Audio(_) => Err(LLMError::InvalidRequest(
            "OpenAI responses API does not accept audio chat messages".to_string(),
        )),
        MessageType::ToolUse(_) | MessageType::ToolResult(_) => Err(LLMError::InvalidRequest(
            "Tool messages are mapped as function_call items".to_string(),
        )),
    }
}

fn text_content_for_role(role: &ChatRole, text: &str) -> ResponsesInputContent {
    match role {
        ChatRole::User => ResponsesInputContent::Text {
            text: text.to_string(),
        },
        ChatRole::Assistant => ResponsesInputContent::OutputText {
            text: text.to_string(),
        },
    }
}

fn map_image_message(
    role: &ChatRole,
    text: &str,
    url: &str,
) -> Result<Vec<ResponsesInputContent>, LLMError> {
    if !matches!(role, ChatRole::User) {
        return Err(LLMError::InvalidRequest(
            "OpenAI responses assistant messages only support output_text or refusal".to_string(),
        ));
    }
    Ok(image_parts(text, url))
}

fn image_parts(text: &str, url: &str) -> Vec<ResponsesInputContent> {
    let mut parts = Vec::new();
    if !text.trim().is_empty() {
        parts.push(ResponsesInputContent::Text {
            text: text.to_string(),
        });
    }
    parts.push(ResponsesInputContent::Image {
        image_url: url.to_string(),
    });
    parts
}

fn image_data_url(mime: &ImageMime, bytes: &[u8]) -> String {
    let encoded = STANDARD.encode(bytes);
    format!("data:{};base64,{}", mime.mime_type(), encoded)
}

fn map_tool_calls(calls: &[ToolCall]) -> Vec<ResponsesInputItem> {
    calls
        .iter()
        .map(|call| {
            ResponsesInputItem::FunctionCall(ResponsesFunctionCallItem {
                item_type: ResponsesFunctionCallItemType::FunctionCall,
                id: None,
                call_id: call.id.clone(),
                name: call.function.name.clone(),
                arguments: call.function.arguments.clone(),
            })
        })
        .collect()
}

fn map_tool_results(results: &[ToolCall]) -> Vec<ResponsesInputItem> {
    results
        .iter()
        .map(|result| {
            ResponsesInputItem::FunctionCallOutput(ResponsesFunctionCallOutputItem {
                item_type: ResponsesFunctionCallOutputItemType::FunctionCallOutput,
                call_id: result.id.clone(),
                output: result.function.arguments.clone(),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests;
