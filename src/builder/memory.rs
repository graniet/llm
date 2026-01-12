use crate::memory::{MemoryProvider, SlidingWindowMemory, TrimStrategy};

use super::llm_builder::LLMBuilder;

impl LLMBuilder {
    /// Sets a custom memory provider for conversation history.
    pub fn memory(mut self, memory: impl MemoryProvider + 'static) -> Self {
        self.state.memory = Some(Box::new(memory));
        self
    }

    /// Sets a sliding window memory provider.
    pub fn sliding_memory(mut self, memory: SlidingWindowMemory) -> Self {
        self.state.memory = Some(Box::new(memory));
        self
    }

    /// Sets up a sliding window memory with the specified window size.
    pub fn sliding_window_memory(mut self, window_size: usize) -> Self {
        self.state.memory = Some(Box::new(SlidingWindowMemory::new(window_size)));
        self
    }

    /// Sets up a sliding window memory with specified trim strategy.
    pub fn sliding_window_with_strategy(
        mut self,
        window_size: usize,
        strategy: TrimStrategy,
    ) -> Self {
        self.state.memory = Some(Box::new(SlidingWindowMemory::with_strategy(
            window_size,
            strategy,
        )));
        self
    }
}
