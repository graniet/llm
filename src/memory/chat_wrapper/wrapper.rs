use std::sync::Arc;

use tokio::sync::RwLock;

use crate::{
    memory::{MemoryProvider, MessageCondition},
    LLMProvider,
};

use super::reactive::{spawn_reactive_listener, ReactiveConfig};

/// Configuration for ChatWithMemory.
pub struct ChatWithMemoryConfig {
    provider: Arc<dyn LLMProvider>,
    memory: Arc<RwLock<Box<dyn MemoryProvider>>>,
    role: Option<String>,
    role_triggers: Vec<(String, MessageCondition)>,
    max_cycles: Option<u32>,
    stt_provider: Option<Arc<dyn LLMProvider>>,
}

impl ChatWithMemoryConfig {
    pub fn new(
        provider: Arc<dyn LLMProvider>,
        memory: Arc<RwLock<Box<dyn MemoryProvider>>>,
    ) -> Self {
        Self {
            provider,
            memory,
            role: None,
            role_triggers: Vec::new(),
            max_cycles: None,
            stt_provider: None,
        }
    }

    pub fn role(mut self, role: impl Into<String>) -> Self {
        self.role = Some(role.into());
        self
    }

    pub fn role_triggers(mut self, triggers: Vec<(String, MessageCondition)>) -> Self {
        self.role_triggers = triggers;
        self
    }

    pub fn max_cycles(mut self, max: u32) -> Self {
        self.max_cycles = Some(max);
        self
    }

    pub fn stt_provider(mut self, provider: Option<Arc<dyn LLMProvider>>) -> Self {
        self.stt_provider = provider;
        self
    }
}

/// Adds transparent long-term memory to any `ChatProvider`.
pub struct ChatWithMemory {
    pub(super) provider: Arc<dyn LLMProvider>,
    pub(super) memory: Arc<RwLock<Box<dyn MemoryProvider>>>,
    pub(super) role: Option<String>,
    pub(super) role_triggers: Vec<(String, MessageCondition)>,
    pub(super) max_cycles: Option<u32>,
    pub(super) cycle_counter: Arc<std::sync::atomic::AtomicU32>,
    pub(super) stt_provider: Option<Arc<dyn LLMProvider>>,
}

impl ChatWithMemory {
    pub fn new(
        provider: Arc<dyn LLMProvider>,
        memory: Arc<RwLock<Box<dyn MemoryProvider>>>,
    ) -> Self {
        Self::with_config(ChatWithMemoryConfig::new(provider, memory))
    }

    pub fn with_config(config: ChatWithMemoryConfig) -> Self {
        let wrapper = Self::from_config(config);
        if !wrapper.role_triggers.is_empty() {
            let reactive = ReactiveConfig::new(wrapper.memory.clone(), wrapper.provider.clone())
                .role_triggers(wrapper.role_triggers.clone())
                .role(wrapper.role.clone())
                .max_cycles(wrapper.max_cycles)
                .cycle_counter(wrapper.cycle_counter.clone());
            spawn_reactive_listener(reactive);
        }
        wrapper
    }

    fn from_config(config: ChatWithMemoryConfig) -> Self {
        Self {
            provider: config.provider,
            memory: config.memory,
            role: config.role,
            role_triggers: config.role_triggers,
            max_cycles: config.max_cycles,
            cycle_counter: Arc::new(std::sync::atomic::AtomicU32::new(0)),
            stt_provider: config.stt_provider,
        }
    }

    /// Access the wrapped provider.
    pub fn inner(&self) -> &dyn LLMProvider {
        self.provider.as_ref()
    }

    /// Dump the full memory (debugging).
    pub async fn memory_contents(&self) -> Vec<crate::chat::ChatMessage> {
        let guard = self.memory.read().await;
        match guard.recall("", None).await {
            Ok(messages) => messages,
            Err(err) => {
                log::warn!("Memory recall error: {err}");
                Vec::new()
            }
        }
    }
}
