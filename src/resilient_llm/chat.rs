use std::pin::Pin;

use async_trait::async_trait;
use futures::stream::Stream;

use crate::{
    chat::{ChatMessage, ChatProvider, ChatResponse, StreamChunk, StreamResponse, Tool},
    error::LLMError,
};

use super::wrapper::ResilientLLM;

#[async_trait]
impl ChatProvider for ResilientLLM {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.retry(|| self.inner.chat_with_tools(messages, tools))
            .await
    }

    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError> {
        self.retry(|| self.inner.chat_stream(messages)).await
    }

    async fn chat_stream_struct(
        &self,
        messages: &[ChatMessage],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>, LLMError>
    {
        self.retry(|| self.inner.chat_stream_struct(messages)).await
    }

    async fn chat_stream_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>, LLMError> {
        self.retry(|| self.inner.chat_stream_with_tools(messages, tools))
            .await
    }
}
