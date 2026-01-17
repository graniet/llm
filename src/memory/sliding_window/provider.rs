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
        let mut messages = self.recent_messages(limit);
        messages.retain(|msg| !msg.has_audio());
        Ok(messages)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryProvider;

    #[tokio::test]
    async fn recall_filters_audio_messages() {
        let mut memory = SlidingWindowMemory::new(10);
        let audio = ChatMessage::user().audio(vec![1]).build();
        let text = ChatMessage::user().content("hi").build();
        memory.remember(&audio).await.expect("remember audio");
        memory.remember(&text).await.expect("remember text");

        let recalled = memory.recall("", None).await.expect("recall");
        assert_eq!(recalled.len(), 1);
        assert_eq!(recalled[0].content, "hi");
        assert!(!recalled[0].has_audio());
    }
}
