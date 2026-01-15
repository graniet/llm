use async_trait::async_trait;

use crate::{chat::ChatMessage, error::LLMError};

use super::core::{SlidingWindowMemory, TrimStrategy};
use crate::memory::{MemoryProvider, MemoryType};

#[async_trait]
impl MemoryProvider for SlidingWindowMemory {
    async fn remember(&mut self, message: &ChatMessage) -> Result<(), LLMError> {
        if self.messages.len() >= self.window_size.get() {
            match self.trim_strategy {
                TrimStrategy::Drop => {
                    self.messages.pop_front();
                }
                TrimStrategy::Summarize => self.mark_for_summary(),
            }
        }
        self.messages.push_back(message.clone());
        Ok(())
    }

    async fn recall(
        &self,
        _query: &str,
        limit: Option<usize>,
    ) -> Result<Vec<ChatMessage>, LLMError> {
        let limit = limit.unwrap_or(self.messages.len());
        Ok(self.recent_messages(limit))
    }

    async fn clear(&mut self) -> Result<(), LLMError> {
        self.messages.clear();
        Ok(())
    }

    fn memory_type(&self) -> MemoryType {
        MemoryType::SlidingWindow
    }

    fn size(&self) -> usize {
        self.messages.len()
    }

    fn needs_summary(&self) -> bool {
        SlidingWindowMemory::needs_summary(self)
    }

    fn mark_for_summary(&mut self) {
        SlidingWindowMemory::mark_for_summary(self);
    }

    fn replace_with_summary(&mut self, summary: String) {
        SlidingWindowMemory::replace_with_summary(self, summary);
    }
}
