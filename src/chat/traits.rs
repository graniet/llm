use std::pin::Pin;

use async_trait::async_trait;
use futures::stream::Stream;

use crate::error::LLMError;
use crate::ToolCall;

use super::message::ChatMessage;
use super::stream::{StreamChunk, StreamResponse};
use super::tool::Tool;
use super::usage::Usage;

pub trait ChatResponse: std::fmt::Debug + std::fmt::Display + Send + Sync {
    fn text(&self) -> Option<String>;
    fn tool_calls(&self) -> Option<Vec<ToolCall>>;
    fn thinking(&self) -> Option<String> {
        None
    }
    fn usage(&self) -> Option<Usage> {
        None
    }
}

/// Trait for providers that support chat-style interactions.
#[async_trait]
pub trait ChatProvider: Sync + Send {
    async fn chat(&self, messages: &[ChatMessage]) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.chat_with_tools(messages, None).await
    }

    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError>;

    async fn chat_with_web_search(
        &self,
        _input: String,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        Err(LLMError::Generic(
            "Web search not supported for this provider".to_string(),
        ))
    }

    async fn chat_stream(
        &self,
        _messages: &[ChatMessage],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError> {
        Err(LLMError::Generic(
            "Streaming not supported for this provider".to_string(),
        ))
    }

    async fn chat_stream_struct(
        &self,
        _messages: &[ChatMessage],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>, LLMError>
    {
        Err(LLMError::Generic(
            "Structured streaming not supported for this provider".to_string(),
        ))
    }

    async fn chat_stream_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>, LLMError> {
        Err(LLMError::Generic(
            "Streaming with tools not supported for this provider".to_string(),
        ))
    }

    async fn memory_contents(&self) -> Option<Vec<ChatMessage>> {
        None
    }

    async fn summarize_history(&self, msgs: &[ChatMessage]) -> Result<String, LLMError> {
        let prompt = build_summary_prompt(msgs);
        let req = [ChatMessage::user().content(prompt).build()];
        let response = self.chat(&req).await?;
        response
            .text()
            .ok_or_else(|| LLMError::Generic("no text in summary response".into()))
    }
}

fn build_summary_prompt(msgs: &[ChatMessage]) -> String {
    let mut lines = Vec::with_capacity(msgs.len());
    for msg in msgs {
        lines.push(format!("{:?}: {}", msg.role, msg.content));
    }
    format!("Summarize in 2-3 sentences:\n{}", lines.join("\n"))
}
