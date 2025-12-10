//! Resilience wrapper providing retry with exponential backoff for LLM providers.
//!
//! This wrapper retries transient failures with exponential backoff and jitter.
//! It does not retry on permanent errors like authentication or invalid requests.
//!
//! # Example
//!
//! ```no_run
//! use llm::builder::{LLMBackend, LLMBuilder};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let llm = LLMBuilder::new()
//!         .backend(LLMBackend::OpenAI)
//!         .api_key(std::env::var("OPENAI_API_KEY").unwrap_or_default())
//!         .model("gpt-4o-mini")
//!         .resilient(true)
//!         .resilient_attempts(3)
//!         .resilient_backoff(200, 2_000)
//!         .build()?;
//!
//!     let msgs = [
//!         llm::chat::ChatMessage::user().content("Say hi succinctly").build(),
//!     ];
//!     let resp = llm.chat(&msgs).await?;
//!     println!("{}", resp);
//!     Ok(())
//! }
//! ```
use std::time::Duration;

use async_trait::async_trait;
// Deterministic jitter only; no RNG
use tokio::time::sleep;

use crate::chat::{ChatMessage, ChatProvider, ChatResponse, StreamChunk, StreamResponse, Tool};
use crate::completion::{CompletionProvider, CompletionRequest, CompletionResponse};
use crate::embedding::EmbeddingProvider;
use crate::error::LLMError;
use crate::models::{ModelListRequest, ModelListResponse, ModelsProvider};
use crate::stt::SpeechToTextProvider;
use crate::tts::TextToSpeechProvider;
use crate::LLMProvider;

/// Configuration for retry and backoff behavior.
#[derive(Clone, Debug)]
pub struct ResilienceConfig {
    /// Maximum number of attempts including the first one
    pub max_attempts: usize,
    /// Initial backoff delay in milliseconds
    pub base_delay_ms: u64,
    /// Maximum backoff delay in milliseconds
    pub max_delay_ms: u64,
    /// Whether to add random jitter to backoff delays
    pub jitter: bool,
}

impl ResilienceConfig {
    /// Creates a default configuration with sane values.
    pub fn defaults() -> Self {
        Self {
            max_attempts: 3,
            base_delay_ms: 200,
            max_delay_ms: 2_000,
            jitter: true,
        }
    }
}

/// Resilient wrapper that retries transient failures using exponential backoff.
pub struct ResilientLLM {
    inner: Box<dyn LLMProvider>,
    cfg: ResilienceConfig,
}

impl ResilientLLM {
    /// Creates a new resilient wrapper around an existing provider.
    pub fn new(inner: Box<dyn LLMProvider>, cfg: ResilienceConfig) -> Self {
        Self { inner, cfg }
    }

    fn is_retryable(err: &LLMError) -> bool {
        match err {
            LLMError::HttpError(_) => true,
            LLMError::ProviderError(_) => true,
            LLMError::ResponseFormatError { .. } => true,
            LLMError::JsonError(_) => true,
            LLMError::Generic(_) => true,
            LLMError::RetryExceeded { .. } => false,
            LLMError::AuthError(_) => false,
            LLMError::InvalidRequest(_) => false,
            LLMError::ToolConfigError(_) => false,
        }
    }

    async fn backoff_sleep(&self, attempt_index: usize) {
        let mut delay = self
            .cfg
            .base_delay_ms
            .saturating_mul(1u64 << attempt_index.min(16));
        delay = delay.min(self.cfg.max_delay_ms);
        if self.cfg.jitter {
            let span = (delay / 2).max(1);
            let jitter = ((attempt_index as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1))
                % span;
            delay = delay.saturating_sub(jitter);
        }
        sleep(Duration::from_millis(delay)).await;
    }

    // no generic retry function; per-method retries inline to keep Send bounds simple
}

impl LLMProvider for ResilientLLM {}

#[async_trait]
impl ChatProvider for ResilientLLM {
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        let mut attempts_left = self.cfg.max_attempts;
        let mut idx = 0usize;
        let mut last_err: Option<LLMError> = None;
        while attempts_left > 0 {
            match self.inner.chat_with_tools(messages, tools).await {
                Ok(v) => return Ok(v),
                Err(e) => {
                    if attempts_left == 1 || !Self::is_retryable(&e) {
                        return Err(e);
                    }
                    last_err = Some(e);
                    self.backoff_sleep(idx).await;
                    attempts_left -= 1;
                    idx += 1;
                }
            }
        }
        Err(LLMError::RetryExceeded {
            attempts: self.cfg.max_attempts,
            last_error: last_err.map(|e| e.to_string()).unwrap_or_default(),
        })
    }

    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
    ) -> Result<
        std::pin::Pin<Box<dyn futures::stream::Stream<Item = Result<String, LLMError>> + Send>>,
        LLMError,
    > {
        let mut attempts_left = self.cfg.max_attempts;
        let mut idx = 0usize;
        let mut last_err: Option<LLMError> = None;
        while attempts_left > 0 {
            match self.inner.chat_stream(messages).await {
                Ok(v) => return Ok(v),
                Err(e) => {
                    if attempts_left == 1 || !Self::is_retryable(&e) {
                        return Err(e);
                    }
                    last_err = Some(e);
                    self.backoff_sleep(idx).await;
                    attempts_left -= 1;
                    idx += 1;
                }
            }
        }
        Err(LLMError::RetryExceeded {
            attempts: self.cfg.max_attempts,
            last_error: last_err.map(|e| e.to_string()).unwrap_or_default(),
        })
    }

    async fn chat_stream_struct(
        &self,
        messages: &[ChatMessage],
    ) -> Result<
        std::pin::Pin<
            Box<dyn futures::stream::Stream<Item = Result<StreamResponse, LLMError>> + Send>,
        >,
        LLMError,
    > {
        let mut attempts_left = self.cfg.max_attempts;
        let mut idx = 0usize;
        let mut last_err: Option<LLMError> = None;
        while attempts_left > 0 {
            match self.inner.chat_stream_struct(messages).await {
                Ok(v) => return Ok(v),
                Err(e) => {
                    if attempts_left == 1 || !Self::is_retryable(&e) {
                        return Err(e);
                    }
                    last_err = Some(e);
                    self.backoff_sleep(idx).await;
                    attempts_left -= 1;
                    idx += 1;
                }
            }
        }
        Err(LLMError::RetryExceeded {
            attempts: self.cfg.max_attempts,
            last_error: last_err.map(|e| e.to_string()).unwrap_or_default(),
        })
    }

    async fn chat_stream_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<
        std::pin::Pin<
            Box<dyn futures::stream::Stream<Item = Result<StreamChunk, LLMError>> + Send>,
        >,
        LLMError,
    > {
        let mut attempts_left = self.cfg.max_attempts;
        let mut idx = 0usize;
        let mut last_err: Option<LLMError> = None;
        while attempts_left > 0 {
            match self.inner.chat_stream_with_tools(messages, tools).await {
                Ok(v) => return Ok(v),
                Err(e) => {
                    if attempts_left == 1 || !Self::is_retryable(&e) {
                        return Err(e);
                    }
                    last_err = Some(e);
                    self.backoff_sleep(idx).await;
                    attempts_left -= 1;
                    idx += 1;
                }
            }
        }
        Err(LLMError::RetryExceeded {
            attempts: self.cfg.max_attempts,
            last_error: last_err.map(|e| e.to_string()).unwrap_or_default(),
        })
    }
}

#[async_trait]
impl CompletionProvider for ResilientLLM {
    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        let mut attempts_left = self.cfg.max_attempts;
        let mut idx = 0usize;
        let mut last_err: Option<LLMError> = None;
        while attempts_left > 0 {
            match self.inner.complete(req).await {
                Ok(v) => return Ok(v),
                Err(e) => {
                    if attempts_left == 1 || !Self::is_retryable(&e) {
                        return Err(e);
                    }
                    last_err = Some(e);
                    self.backoff_sleep(idx).await;
                    attempts_left -= 1;
                    idx += 1;
                }
            }
        }
        Err(LLMError::RetryExceeded {
            attempts: self.cfg.max_attempts,
            last_error: last_err.map(|e| e.to_string()).unwrap_or_default(),
        })
    }
}

#[async_trait]
impl EmbeddingProvider for ResilientLLM {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        let mut attempts_left = self.cfg.max_attempts;
        let mut idx = 0usize;
        let mut last_err: Option<LLMError> = None;
        while attempts_left > 0 {
            match self.inner.embed(input.clone()).await {
                Ok(v) => return Ok(v),
                Err(e) => {
                    if attempts_left == 1 || !Self::is_retryable(&e) {
                        return Err(e);
                    }
                    last_err = Some(e);
                    self.backoff_sleep(idx).await;
                    attempts_left -= 1;
                    idx += 1;
                }
            }
        }
        Err(LLMError::RetryExceeded {
            attempts: self.cfg.max_attempts,
            last_error: last_err.map(|e| e.to_string()).unwrap_or_default(),
        })
    }
}

#[async_trait]
impl SpeechToTextProvider for ResilientLLM {
    async fn transcribe(&self, audio: Vec<u8>) -> Result<String, LLMError> {
        let mut attempts_left = self.cfg.max_attempts;
        let mut idx = 0usize;
        let mut last_err: Option<LLMError> = None;
        while attempts_left > 0 {
            match self.inner.transcribe(audio.clone()).await {
                Ok(v) => return Ok(v),
                Err(e) => {
                    if attempts_left == 1 || !Self::is_retryable(&e) {
                        return Err(e);
                    }
                    last_err = Some(e);
                    self.backoff_sleep(idx).await;
                    attempts_left -= 1;
                    idx += 1;
                }
            }
        }
        Err(LLMError::RetryExceeded {
            attempts: self.cfg.max_attempts,
            last_error: last_err.map(|e| e.to_string()).unwrap_or_default(),
        })
    }
}

#[async_trait]
impl TextToSpeechProvider for ResilientLLM {
    async fn speech(&self, text: &str) -> Result<Vec<u8>, LLMError> {
        let mut attempts_left = self.cfg.max_attempts;
        let mut idx = 0usize;
        let mut last_err: Option<LLMError> = None;
        let text_owned = text.to_owned();
        while attempts_left > 0 {
            match self.inner.speech(&text_owned).await {
                Ok(v) => return Ok(v),
                Err(e) => {
                    if attempts_left == 1 || !Self::is_retryable(&e) {
                        return Err(e);
                    }
                    last_err = Some(e);
                    self.backoff_sleep(idx).await;
                    attempts_left -= 1;
                    idx += 1;
                }
            }
        }
        Err(LLMError::RetryExceeded {
            attempts: self.cfg.max_attempts,
            last_error: last_err.map(|e| e.to_string()).unwrap_or_default(),
        })
    }
}

#[async_trait]
impl ModelsProvider for ResilientLLM {
    async fn list_models(
        &self,
        request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        let mut attempts_left = self.cfg.max_attempts;
        let mut idx = 0usize;
        let mut last_err: Option<LLMError> = None;
        while attempts_left > 0 {
            match self.inner.list_models(request).await {
                Ok(v) => return Ok(v),
                Err(e) => {
                    if attempts_left == 1 || !Self::is_retryable(&e) {
                        return Err(e);
                    }
                    last_err = Some(e);
                    self.backoff_sleep(idx).await;
                    attempts_left -= 1;
                    idx += 1;
                }
            }
        }
        Err(LLMError::RetryExceeded {
            attempts: self.cfg.max_attempts,
            last_error: last_err.map(|e| e.to_string()).unwrap_or_default(),
        })
    }
}
