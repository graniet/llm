//! LLM (Rust LLM) is a unified interface for interacting with Large Language Model providers.
//!
//! # Overview
//! This crate provides a consistent API for working with different LLM backends by abstracting away
//! provider-specific implementation details. It supports:
//!
//! - Chat-based interactions
//! - Text completion
//! - Embeddings generation
//! - Multiple providers (OpenAI, Anthropic, etc.)
//! - Request validation and retry logic
//!
//! # Architecture
//! The crate is organized into modules that handle different aspects of LLM interactions:

use chat::Tool;
use serde::{Deserialize, Serialize};

/// Backend implementations for supported LLM providers like OpenAI, Anthropic, etc.
pub mod backends;

/// Builder pattern for configuring and instantiating LLM providers
pub mod builder;

/// Chain multiple LLM providers together for complex workflows
pub mod chain;

/// Chat-based interactions with language models (e.g. ChatGPT style)
pub mod chat;

/// Text completion capabilities (e.g. GPT-3 style completion)
pub mod completion;

/// Vector embeddings generation for text
pub mod embedding;

/// Error types and handling
pub mod error;

/// Validation wrapper for LLM providers with retry capabilities
pub mod validated_llm;

/// Evaluator for LLM providers
pub mod evaluator;

/// Core trait that all LLM providers must implement, combining chat, completion
/// and embedding capabilities into a unified interface
pub trait LLMProvider:
    chat::ChatProvider + completion::CompletionProvider + embedding::EmbeddingProvider
{
    fn tools(&self) -> Option<&[Tool]> {
        None
    }
}

/// Tool call from OpenAI's API.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ToolCall {
    /// The ID of the tool call.
    pub id: String,
    /// The type of the tool call.
    #[serde(rename = "type")]
    pub call_type: String,
    /// The function to call.
    pub function: FunctionCall,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct FunctionCall {
    /// The name of the function to call.
    pub name: String,
    /// The arguments to pass to the function.
    pub arguments: String,
}
