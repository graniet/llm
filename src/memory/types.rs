use std::sync::Arc;

use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

use crate::{chat::ChatMessage, error::LLMError};

/// Event emitted when a message is added to reactive memory
#[derive(Debug, Clone)]
pub struct MessageEvent {
    /// Role of the agent that sent the message
    pub role: String,
    /// The chat message content
    pub msg: ChatMessage,
}

/// Conditions for triggering reactive message handlers
#[derive(Clone)]
pub enum MessageCondition {
    /// Always trigger
    Any,
    /// Trigger if message content equals exact string
    Eq(String),
    /// Trigger if message content contains substring
    Contains(String),
    /// Trigger if message content does not contain substring
    NotContains(String),
    /// Trigger if sender role matches
    RoleIs(String),
    /// Trigger if sender role does not match
    RoleNot(String),
    /// Trigger if message length is greater than specified
    LenGt(usize),
    /// Custom condition function
    Custom(Arc<dyn Fn(&ChatMessage) -> bool + Send + Sync>),
    /// Empty
    Empty,
    /// Trigger if all conditions are met
    All(Vec<MessageCondition>),
    /// Trigger if any condition is met
    AnyOf(Vec<MessageCondition>),
    /// Trigger if message content matches regex
    Regex(String),
}

impl MessageCondition {
    /// Check if the condition is met for the given message event
    pub fn matches(&self, event: &MessageEvent) -> bool {
        match self {
            MessageCondition::Any => true,
            MessageCondition::Eq(text) => event.msg.content == *text,
            MessageCondition::Contains(text) => event.msg.content.contains(text),
            MessageCondition::NotContains(text) => !event.msg.content.contains(text),
            MessageCondition::RoleIs(role) => event.role == *role,
            MessageCondition::RoleNot(role) => event.role != *role,
            MessageCondition::LenGt(len) => event.msg.content.len() > *len,
            MessageCondition::Custom(func) => func(&event.msg),
            MessageCondition::Empty => event.msg.content.is_empty(),
            MessageCondition::All(inner) => inner.iter().all(|c| c.matches(event)),
            MessageCondition::AnyOf(inner) => inner.iter().any(|c| c.matches(event)),
            MessageCondition::Regex(regex) => Regex::new(regex)
                .map(|re| re.is_match(&event.msg.content))
                .unwrap_or(false),
        }
    }
}

/// Types of memory implementations available
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryType {
    /// Simple sliding window that keeps the N most recent messages
    SlidingWindow,
}

/// Trait for memory providers that can store and retrieve conversation history.
#[async_trait]
pub trait MemoryProvider: Send + Sync {
    async fn remember(&mut self, message: &ChatMessage) -> Result<(), LLMError>;

    async fn recall(&self, query: &str, limit: Option<usize>)
        -> Result<Vec<ChatMessage>, LLMError>;

    async fn clear(&mut self) -> Result<(), LLMError>;

    fn memory_type(&self) -> MemoryType;

    fn size(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.size() == 0
    }

    fn needs_summary(&self) -> bool {
        false
    }

    fn mark_for_summary(&mut self) {}

    fn replace_with_summary(&mut self, _summary: String) {}

    fn get_event_receiver(&self) -> Option<broadcast::Receiver<MessageEvent>> {
        None
    }

    async fn remember_with_role(
        &mut self,
        message: &ChatMessage,
        _role: String,
    ) -> Result<(), LLMError> {
        self.remember(message).await
    }
}
