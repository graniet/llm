//! Metrics provider wrapper for timing non-streaming chat calls.
//!
//! This module provides [`MetricsProvider`], a wrapper that adds timing
//! information to chat responses when metrics collection is enabled.

use std::fmt;
use std::pin::Pin;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::Stream;

use crate::chat::{
    ChatMessage, ChatMetrics, ChatProvider, ChatResponse, StreamChunk, StreamResponse, Tool,
    Usage,
};
use crate::completion::{CompletionProvider, CompletionRequest, CompletionResponse};
use crate::embedding::EmbeddingProvider;
use crate::error::LLMError;
use crate::models::{ModelListRequest, ModelListResponse as ModelListResponseTrait, ModelsProvider};
use crate::stt::SpeechToTextProvider;
use crate::tts::TextToSpeechProvider;
use crate::{LLMProvider, ToolCall};

/// A provider wrapper that adds timing metrics to chat responses.
///
/// This wrapper intercepts `chat_with_tools` calls to measure duration
/// and wraps the response to include metrics. All other methods are
/// delegated directly to the inner provider.
///
/// Created automatically by the builder when `.enable_metrics(true)` is set.
pub struct MetricsProvider {
    inner: Box<dyn LLMProvider>,
}

impl MetricsProvider {
    /// Create a new metrics-enabled provider wrapper.
    pub fn new(inner: Box<dyn LLMProvider>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl ChatProvider for MetricsProvider {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        let start = Instant::now();
        let response = self.inner.chat_with_tools(messages, tools).await?;
        let duration = start.elapsed();

        Ok(Box::new(MetricsResponse {
            inner: response,
            duration,
        }))
    }

    async fn chat_with_web_search(
        &self,
        input: String,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        let start = Instant::now();
        let response = self.inner.chat_with_web_search(input).await?;
        let duration = start.elapsed();

        Ok(Box::new(MetricsResponse {
            inner: response,
            duration,
        }))
    }

    // Streaming methods are delegated - user should use Tracked wrapper
    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError> {
        self.inner.chat_stream(messages).await
    }

    async fn chat_stream_struct(
        &self,
        messages: &[ChatMessage],
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamResponse, LLMError>> + Send>>, LLMError>
    {
        self.inner.chat_stream_struct(messages).await
    }

    async fn chat_stream_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, LLMError>> + Send>>, LLMError> {
        self.inner.chat_stream_with_tools(messages, tools).await
    }
}

#[async_trait]
impl CompletionProvider for MetricsProvider {
    async fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        self.inner.complete(request).await
    }
}

#[async_trait]
impl EmbeddingProvider for MetricsProvider {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        self.inner.embed(input).await
    }
}

#[async_trait]
impl SpeechToTextProvider for MetricsProvider {
    async fn transcribe(&self, audio_data: Vec<u8>) -> Result<String, LLMError> {
        self.inner.transcribe(audio_data).await
    }
}

#[async_trait]
impl TextToSpeechProvider for MetricsProvider {
    async fn speech(&self, input: &str) -> Result<Vec<u8>, LLMError> {
        self.inner.speech(input).await
    }
}

#[async_trait]
impl ModelsProvider for MetricsProvider {
    async fn list_models(
        &self,
        request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponseTrait>, LLMError> {
        self.inner.list_models(request).await
    }
}

impl LLMProvider for MetricsProvider {
    fn tools(&self) -> Option<&[Tool]> {
        self.inner.tools()
    }
}

/// A chat response wrapper that includes timing metrics.
struct MetricsResponse {
    inner: Box<dyn ChatResponse>,
    duration: Duration,
}

impl ChatResponse for MetricsResponse {
    fn text(&self) -> Option<String> {
        self.inner.text()
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        self.inner.tool_calls()
    }

    fn thinking(&self) -> Option<String> {
        self.inner.thinking()
    }

    fn usage(&self) -> Option<Usage> {
        self.inner.usage()
    }

    fn metrics(&self) -> Option<ChatMetrics> {
        Some(ChatMetrics {
            usage: self.inner.usage(),
            duration: self.duration,
            time_to_first_token: None, // N/A for non-streaming
        })
    }
}

impl fmt::Debug for MetricsResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MetricsResponse")
            .field("inner", &self.inner)
            .field("duration", &self.duration)
            .finish()
    }
}

impl fmt::Display for MetricsResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock provider for testing
    struct MockProvider;

    struct MockResponse {
        text: String,
        usage: Option<Usage>,
    }

    impl fmt::Debug for MockResponse {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("MockResponse")
                .field("text", &self.text)
                .finish()
        }
    }

    impl fmt::Display for MockResponse {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}", self.text)
        }
    }

    impl ChatResponse for MockResponse {
        fn text(&self) -> Option<String> {
            Some(self.text.clone())
        }

        fn tool_calls(&self) -> Option<Vec<ToolCall>> {
            None
        }

        fn usage(&self) -> Option<Usage> {
            self.usage.clone()
        }
    }

    #[async_trait]
    impl ChatProvider for MockProvider {
        async fn chat_with_tools(
            &self,
            _messages: &[ChatMessage],
            _tools: Option<&[Tool]>,
        ) -> Result<Box<dyn ChatResponse>, LLMError> {
            // Simulate some work
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            Ok(Box::new(MockResponse {
                text: "Hello".to_string(),
                usage: Some(Usage {
                    prompt_tokens: 10,
                    completion_tokens: 5,
                    total_tokens: 15,
                    completion_tokens_details: None,
                    prompt_tokens_details: None,
                }),
            }))
        }
    }

    #[async_trait]
    impl CompletionProvider for MockProvider {
        async fn complete(
            &self,
            _request: &CompletionRequest,
        ) -> Result<CompletionResponse, LLMError> {
            Err(LLMError::Generic("Not implemented".to_string()))
        }
    }

    #[async_trait]
    impl EmbeddingProvider for MockProvider {
        async fn embed(&self, _input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
            Err(LLMError::Generic("Not implemented".to_string()))
        }
    }

    #[async_trait]
    impl SpeechToTextProvider for MockProvider {
        async fn transcribe(&self, _audio_data: Vec<u8>) -> Result<String, LLMError> {
            Err(LLMError::Generic("Not implemented".to_string()))
        }
    }

    #[async_trait]
    impl TextToSpeechProvider for MockProvider {
        async fn speech(&self, _input: &str) -> Result<Vec<u8>, LLMError> {
            Err(LLMError::Generic("Not implemented".to_string()))
        }
    }

    #[async_trait]
    impl ModelsProvider for MockProvider {
        async fn list_models(
            &self,
            _request: Option<&ModelListRequest>,
        ) -> Result<Box<dyn ModelListResponseTrait>, LLMError> {
            Err(LLMError::Generic("Not implemented".to_string()))
        }
    }

    impl LLMProvider for MockProvider {}

    #[tokio::test]
    async fn test_metrics_provider_adds_timing() {
        let provider = MetricsProvider::new(Box::new(MockProvider));
        let messages = vec![];

        let response = provider.chat_with_tools(&messages, None).await.unwrap();

        // Should have metrics
        let metrics = response.metrics().unwrap();
        assert!(metrics.duration.as_millis() >= 10);
        assert!(metrics.time_to_first_token.is_none()); // Non-streaming

        // Should preserve original response
        assert_eq!(response.text(), Some("Hello".to_string()));
        assert!(response.usage().is_some());
    }

    #[tokio::test]
    async fn test_metrics_response_includes_usage() {
        let provider = MetricsProvider::new(Box::new(MockProvider));
        let messages = vec![];

        let response = provider.chat_with_tools(&messages, None).await.unwrap();
        let metrics = response.metrics().unwrap();

        // Usage should be passed through
        let usage = metrics.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 10);
        assert_eq!(usage.completion_tokens, 5);
        assert_eq!(usage.total_tokens, 15);
    }

    #[test]
    fn test_metrics_response_display() {
        let response = MetricsResponse {
            inner: Box::new(MockResponse {
                text: "Test".to_string(),
                usage: None,
            }),
            duration: Duration::from_millis(100),
        };

        assert_eq!(format!("{}", response), "Test");
    }
}
