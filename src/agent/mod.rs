//! Agent module for building reactive, memory-enabled LLM agents.

#[cfg(feature = "agent")]
pub mod builder;

#[cfg(feature = "agent")]
pub use builder::AgentBuilder;
