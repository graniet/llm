//! Multi-LLM dialogue module.
//!
//! This module provides the ability to have multiple LLM participants
//! engage in a dialogue, with configurable modes (collaboration, debate,
//! critique, free-form) and dynamic participant management.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

mod controller;
mod events;
mod participant;

pub use controller::{DialogueController, TurnMode};
pub use events::StopReason;

// Public API items for future use
#[allow(unused_imports)]
pub use controller::DialogueState;
#[allow(unused_imports)]
pub use events::DialogueEvent;
pub use participant::{DialogueParticipant, ParticipantColor, ParticipantId, ParticipantParams};

/// Mode of dialogue between participants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum DialogueMode {
    /// Collaboration: LLMs work together toward a common goal.
    Collaboration,
    /// Debate: LLMs defend opposing positions.
    Debate,
    /// Critique: One LLM generates, others critique and improve.
    Critique,
    /// Free-form: Open conversation without constraints.
    #[default]
    FreeForm,
    /// Custom: User-defined behavior via system prompts.
    Custom,
}

impl DialogueMode {
    /// Returns the display name for the mode.
    #[must_use]
    pub const fn display_name(&self) -> &'static str {
        match self {
            Self::Collaboration => "Collaboration",
            Self::Debate => "Debate",
            Self::Critique => "Critique",
            Self::FreeForm => "Free",
            Self::Custom => "Custom",
        }
    }

    /// Returns a short description of the mode.
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::Collaboration => "LLMs work together toward a common goal",
            Self::Debate => "LLMs defend opposing positions",
            Self::Critique => "One generates, others critique and improve",
            Self::FreeForm => "Open conversation without constraints",
            Self::Custom => "User-defined behavior via system prompts",
        }
    }
}

/// Configuration for a dialogue session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueConfig {
    /// Mode of dialogue.
    pub mode: DialogueMode,
    /// Initial prompt to start the dialogue.
    pub initial_prompt: String,
    /// Turn mode for participant ordering.
    pub turn_mode: TurnMode,
}

impl Default for DialogueConfig {
    fn default() -> Self {
        Self {
            mode: DialogueMode::FreeForm,
            initial_prompt: String::new(),
            turn_mode: TurnMode::RoundRobin,
        }
    }
}
