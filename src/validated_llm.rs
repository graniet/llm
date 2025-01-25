//! A module providing validation capabilities for LLM responses through a wrapper implementation.
//!
//! This module enables adding custom validation logic to any LLM provider by wrapping it in a
//! `ValidatedLLM` struct. The wrapper will validate responses and retry failed attempts with
//! feedback to help guide the model toward producing valid output.
//!
//! # Example
//!
//! ```no_run
//! use llm::{LLMBuilder, LLMBackend};
//!
//! let llm = LLMBuilder::new()
//!     .backend(LLMBackend::OpenAI)
//!     .validator(|response| {
//!         if response.contains("unsafe content") {
//!             Err("Response contains unsafe content".to_string())
//!         } else {
//!             Ok(())
//!         }
//!     })
//!     .validator_attempts(3)
//!     .build()
//!     .unwrap();
//! ```

use crate::chat::{ChatMessage, ChatProvider, ChatResponse, ChatRole, Tool};
use crate::completion::{CompletionProvider, CompletionRequest, CompletionResponse};
use crate::embedding::EmbeddingProvider;
use crate::error::LLMError;
use crate::{builder::ValidatorFn, LLMProvider};

/// A wrapper around an LLM provider that validates responses before returning them.
///
/// The wrapper implements validation by:
/// 1. Sending the request to the underlying provider
/// 2. Validating the response using the provided validator function
/// 3. If validation fails, retrying with feedback up to the configured number of attempts
///
/// # Type Parameters
///
/// The wrapped provider must implement the `LLMProvider` trait.
pub struct ValidatedLLM {
    /// The wrapped LLM provider
    inner: Box<dyn LLMProvider>,
    /// Function used to validate responses, returns Ok(()) if valid or Err with message if invalid
    validator: Box<ValidatorFn>,
    /// Maximum number of validation attempts before giving up
    attempts: usize,
}

impl ValidatedLLM {
    /// Creates a new ValidatedLLM wrapper around an existing LLM provider.
    ///
    /// # Arguments
    ///
    /// * `inner` - The LLM provider to wrap with validation
    /// * `validator` - Function that takes a response string and returns Ok(()) if valid,
    ///                or Err with error message if invalid
    /// * `attempts` - Maximum number of validation attempts before failing
    ///
    /// # Returns
    ///
    /// A new ValidatedLLM instance configured with the provided parameters.
    pub fn new(inner: Box<dyn LLMProvider>, validator: Box<ValidatorFn>, attempts: usize) -> Self {
        Self {
            inner,
            validator,
            attempts,
        }
    }
}

impl LLMProvider for ValidatedLLM {}

impl ChatProvider for ValidatedLLM {
    /// Sends a chat request and validates the response.
    ///
    /// If validation fails, retries with feedback to the model about the validation error.
    /// The feedback is appended as a new user message to help guide the model.
    ///
    /// # Arguments
    ///
    /// * `messages` - The chat messages to send to the model
    ///
    /// # Returns
    ///
    /// * `Ok(String)` - The validated response from the model
    /// * `Err(LLMError)` - If validation fails after max attempts or other errors occur
    fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        let mut local_messages = messages.to_vec();
        let mut remaining_attempts = self.attempts;

        loop {
            let response = match self.inner.chat_with_tools(&local_messages, tools) {
                Ok(resp) => resp,
                Err(e) => return Err(e),
            };

            match (self.validator)(
                response
                    .texts()
                    .unwrap_or_default()
                    .first()
                    .unwrap_or(&String::new()),
            ) {
                Ok(()) => {
                    return Ok(response);
                }
                Err(err) => {
                    remaining_attempts -= 1;
                    if remaining_attempts == 0 {
                        return Err(LLMError::InvalidRequest(format!(
                            "Validation error after max attempts: {}",
                            err
                        )));
                    }

                    local_messages.push(ChatMessage {
                        role: ChatRole::User,
                        content: format!(
                            "Your previous output was invalid because: {}\n\
                             Please try again and produce a valid response.",
                            err
                        ),
                    });
                }
            }
        }
    }
}

impl CompletionProvider for ValidatedLLM {
    /// Sends a completion request and validates the response.
    ///
    /// If validation fails, retries up to the configured number of attempts.
    /// Unlike chat, completion requests don't support adding feedback messages.
    ///
    /// # Arguments
    ///
    /// * `req` - The completion request to send
    ///
    /// # Returns
    ///
    /// * `Ok(CompletionResponse)` - The validated completion response
    /// * `Err(LLMError)` - If validation fails after max attempts or other errors occur
    fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        let mut remaining_attempts = self.attempts;

        loop {
            let response = match self.inner.complete(req) {
                Ok(resp) => resp,
                Err(e) => return Err(e),
            };

            match (self.validator)(&response.text) {
                Ok(()) => {
                    return Ok(response);
                }
                Err(err) => {
                    remaining_attempts -= 1;
                    if remaining_attempts == 0 {
                        return Err(LLMError::InvalidRequest(format!(
                            "Validation error after max attempts: {}",
                            err
                        )));
                    }
                }
            }
        }
    }
}

impl EmbeddingProvider for ValidatedLLM {
    /// Passes through embedding requests to the inner provider without validation.
    ///
    /// Embeddings are numerical vectors that represent text semantically and don't
    /// require validation since they're not human-readable content.
    ///
    /// # Arguments
    ///
    /// * `input` - Vector of strings to generate embeddings for
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<Vec<f32>>)` - Vector of embedding vectors
    /// * `Err(LLMError)` - If the embedding generation fails
    fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        // Pass through to inner provider since embeddings don't need validation
        self.inner.embed(input)
    }
}
