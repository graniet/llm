//! Agent builder for creating reactive LLM agents with memory and role-based triggers.

use crate::{
    builder::LLMBuilder,
    error::LLMError,
    memory::{ChatWithMemory, MemoryProvider, MessageCondition},
    LLMProvider,
};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Builder for creating reactive LLM agents.
///
/// AgentBuilder provides a clean interface for creating agents that can:
/// - React to messages from other agents based on role and conditions
/// - Share memory across multiple agents
/// - Control reactive cycles to prevent infinite loops
/// - Maintain conversation context
/// - Handle speech-to-text and text-to-speech capabilities
pub struct AgentBuilder {
    llm_builder: LLMBuilder,
    role: Option<String>,
    role_triggers: Vec<(String, MessageCondition)>,
    max_cycles: Option<u32>,
    single_reply_per_turn: bool,
    debounce_ms: Option<u64>,
    stt_builder: Option<LLMBuilder>,
    tts_builder: Option<LLMBuilder>,
    memory: Option<Box<dyn MemoryProvider>>,
}

impl AgentBuilder {
    /// Creates a new AgentBuilder instance.
    pub fn new() -> Self {
        Self {
            llm_builder: LLMBuilder::new(),
            role: None,
            role_triggers: Vec::new(),
            max_cycles: None,
            single_reply_per_turn: false,
            debounce_ms: None,
            stt_builder: None,
            tts_builder: None,
            memory: None,
        }
    }

    /// Sets the role name for this agent.
    ///
    /// The role is used to identify messages from this agent in shared memory
    /// and for reactive message filtering.
    pub fn role(mut self, role: impl Into<String>) -> Self {
        self.role = Some(role.into());
        self
    }

    /// Configures the agent to react to messages from a specific role with a condition.
    ///
    /// The agent will only trigger when messages from the specified role match the condition.
    pub fn on(mut self, role: impl Into<String>, condition: MessageCondition) -> Self {
        self.role_triggers.push((role.into(), condition));
        self
    }

    /// Sets the maximum number of reactive cycles this agent can perform.
    ///
    /// This prevents infinite loops in multi-agent conversations.
    pub fn max_cycles(mut self, max: u32) -> Self {
        self.max_cycles = Some(max);
        self
    }

    /// Configures the agent to send only one reply per conversational turn.
    pub fn single_reply_per_turn(mut self, enabled: bool) -> Self {
        self.single_reply_per_turn = enabled;
        self
    }

    /// Sets a debounce delay in milliseconds before reacting to messages.
    pub fn debounce(mut self, ms: u64) -> Self {
        self.debounce_ms = Some(ms);
        self
    }

    /// Sets the underlying LLM configuration.
    pub fn llm(mut self, llm_builder: LLMBuilder) -> Self {
        self.llm_builder = llm_builder;
        self
    }

    /// Sets the Speech-to-Text LLM configuration.
    pub fn stt(mut self, stt_builder: LLMBuilder) -> Self {
        self.stt_builder = Some(stt_builder);
        self
    }

    /// Sets the Text-to-Speech LLM configuration.
    pub fn tts(mut self, tts_builder: LLMBuilder) -> Self {
        self.tts_builder = Some(tts_builder);
        self
    }

    /// Sets a memory provider for the agent.
    pub fn memory(mut self, memory: impl MemoryProvider + 'static) -> Self {
        self.memory = Some(Box::new(memory));
        self
    }

    /// Builds the agent and returns an LLM provider with agent capabilities.
    ///
    /// # Errors
    ///
    /// Returns an error if the underlying LLM configuration is invalid.
    pub fn build(self) -> Result<Box<dyn LLMProvider>, LLMError> {
        // Build the base LLM provider
        let base_provider = self.llm_builder.build()?;

        // If memory is configured, wrap with ChatWithMemory including agent capabilities
        if let Some(memory) = self.memory {
            let memory_arc = Arc::new(RwLock::new(memory));
            let provider_arc = Arc::from(base_provider);
            let agent_provider = ChatWithMemory::new(
                provider_arc,
                memory_arc,
                self.role,
                self.role_triggers,
                self.max_cycles,
            );
            Ok(Box::new(agent_provider))
        } else {
            // No memory, return base provider
            Ok(base_provider)
        }
    }
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}
