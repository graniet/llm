use std::{collections::VecDeque, num::NonZeroUsize};

use crate::{chat::ChatMessage, error::LLMError};

/// Strategy for handling memory when window size limit is reached
#[derive(Debug, Clone)]
pub enum TrimStrategy {
    /// Drop oldest messages (FIFO behavior)
    Drop,
    /// Summarize all messages into one before adding new ones
    Summarize,
}

/// Non-zero window size for sliding memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowSize(NonZeroUsize);

impl WindowSize {
    pub fn get(self) -> usize {
        self.0.get()
    }

    fn fallback() -> Self {
        // SAFETY: constant is non-zero.
        Self(NonZeroUsize::new(1).expect("non-zero window size"))
    }
}

impl TryFrom<usize> for WindowSize {
    type Error = LLMError;

    fn try_from(value: usize) -> Result<Self, Self::Error> {
        let Some(size) = NonZeroUsize::new(value) else {
            return Err(LLMError::InvalidRequest(
                "Window size must be greater than 0".to_string(),
            ));
        };
        Ok(Self(size))
    }
}

/// Simple sliding window memory that keeps the N most recent messages.
#[derive(Debug, Clone)]
pub struct SlidingWindowMemory {
    pub(super) messages: VecDeque<ChatMessage>,
    pub(super) window_size: WindowSize,
    pub(super) trim_strategy: TrimStrategy,
    pub(super) needs_summary: bool,
}

impl SlidingWindowMemory {
    /// Create a new sliding window memory with the specified window size.
    pub fn new(window_size: usize) -> Self {
        Self::with_strategy(window_size, TrimStrategy::Drop)
    }

    /// Create a new sliding window memory with specified trim strategy.
    pub fn with_strategy(window_size: usize, strategy: TrimStrategy) -> Self {
        let window_size = normalize_window_size(window_size);
        Self::with_window_size(window_size, strategy)
    }

    /// Create a new sliding window memory with validation.
    pub fn try_new(window_size: usize) -> Result<Self, LLMError> {
        let window_size = WindowSize::try_from(window_size)?;
        Ok(Self::with_window_size(window_size, TrimStrategy::Drop))
    }

    /// Create a new sliding window memory with validation and strategy.
    pub fn try_with_strategy(window_size: usize, strategy: TrimStrategy) -> Result<Self, LLMError> {
        let window_size = WindowSize::try_from(window_size)?;
        Ok(Self::with_window_size(window_size, strategy))
    }

    fn with_window_size(window_size: WindowSize, strategy: TrimStrategy) -> Self {
        Self {
            messages: VecDeque::with_capacity(window_size.get()),
            window_size,
            trim_strategy: strategy,
            needs_summary: false,
        }
    }

    /// Get the configured window size.
    pub fn window_size(&self) -> usize {
        self.window_size.get()
    }

    /// Get all stored messages in chronological order.
    pub fn messages(&self) -> Vec<ChatMessage> {
        self.messages.iter().cloned().collect()
    }

    /// Get the most recent N messages.
    pub fn recent_messages(&self, limit: usize) -> Vec<ChatMessage> {
        let len = self.messages.len();
        let start = len.saturating_sub(limit);
        self.messages.range(start..).cloned().collect()
    }

    /// Check if memory needs summarization.
    pub fn needs_summary(&self) -> bool {
        self.needs_summary
    }

    /// Mark memory as needing summarization.
    pub fn mark_for_summary(&mut self) {
        self.needs_summary = true;
    }

    /// Replace all messages with a summary.
    pub fn replace_with_summary(&mut self, summary: String) {
        self.messages.clear();
        self.messages
            .push_back(ChatMessage::assistant().content(summary).build());
        self.needs_summary = false;
    }
}

fn normalize_window_size(window_size: usize) -> WindowSize {
    match WindowSize::try_from(window_size) {
        Ok(size) => size,
        Err(err) => {
            log::warn!("Invalid window size: {err}");
            WindowSize::fallback()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{SlidingWindowMemory, WindowSize};

    #[test]
    fn window_size_rejects_zero() {
        assert!(WindowSize::try_from(0).is_err());
    }

    #[test]
    fn sliding_window_new_falls_back_to_one() {
        let memory = SlidingWindowMemory::new(0);
        assert_eq!(memory.window_size(), 1);
    }

    #[test]
    fn sliding_window_try_new_rejects_zero() {
        assert!(SlidingWindowMemory::try_new(0).is_err());
    }
}
