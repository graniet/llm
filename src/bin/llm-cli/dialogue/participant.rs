//! Dialogue participant types and configuration.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use llm::LLMProvider;

/// Unique identifier for a dialogue participant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ParticipantId(pub u64);

impl ParticipantId {
    /// Creates a new unique participant ID.
    #[must_use]
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for ParticipantId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ParticipantId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "P{}", self.0)
    }
}

/// Color for participant display in UI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ParticipantColor {
    #[default]
    Orange,
    Cyan,
    Green,
    Magenta,
    Yellow,
    Blue,
    Red,
    White,
}

impl ParticipantColor {
    /// Returns the RGB color values.
    #[must_use]
    pub const fn rgb(&self) -> (u8, u8, u8) {
        match self {
            Self::Orange => (217, 119, 87),
            Self::Cyan => (80, 180, 200),
            Self::Green => (120, 200, 140),
            Self::Magenta => (200, 120, 180),
            Self::Yellow => (233, 182, 89),
            Self::Blue => (100, 150, 220),
            Self::Red => (220, 100, 100),
            Self::White => (220, 220, 220),
        }
    }

    /// Returns the color for a given participant index.
    #[must_use]
    pub const fn for_index(index: usize) -> Self {
        const COLORS: [ParticipantColor; 8] = [
            ParticipantColor::Orange,
            ParticipantColor::Cyan,
            ParticipantColor::Green,
            ParticipantColor::Magenta,
            ParticipantColor::Yellow,
            ParticipantColor::Blue,
            ParticipantColor::Red,
            ParticipantColor::White,
        ];
        COLORS[index % COLORS.len()]
    }
}

/// LLM parameters for a participant.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ParticipantParams {
    /// Temperature (0.0 - 2.0).
    pub temperature: Option<f32>,
    /// Maximum tokens to generate.
    pub max_tokens: Option<u32>,
    /// Top P (nucleus sampling).
    pub top_p: Option<f32>,
    /// Presence penalty (-2.0 to 2.0).
    pub presence_penalty: Option<f32>,
    /// Frequency penalty (-2.0 to 2.0).
    pub frequency_penalty: Option<f32>,
    /// Stop sequences.
    #[serde(default)]
    pub stop_sequences: Vec<String>,
}

impl ParticipantParams {
    /// Creates a new builder for participant params.
    #[must_use]
    pub fn builder() -> ParticipantParamsBuilder {
        ParticipantParamsBuilder::default()
    }
}

/// Builder for participant parameters.
#[derive(Debug, Default)]
pub struct ParticipantParamsBuilder {
    params: ParticipantParams,
}

impl ParticipantParamsBuilder {
    /// Sets the temperature.
    #[must_use]
    pub fn temperature(mut self, temp: f32) -> Self {
        self.params.temperature = Some(temp);
        self
    }

    /// Sets the max tokens.
    #[must_use]
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.params.max_tokens = Some(tokens);
        self
    }

    /// Sets the top P.
    #[must_use]
    pub fn top_p(mut self, top_p: f32) -> Self {
        self.params.top_p = Some(top_p);
        self
    }

    /// Sets the presence penalty.
    #[must_use]
    pub fn presence_penalty(mut self, penalty: f32) -> Self {
        self.params.presence_penalty = Some(penalty);
        self
    }

    /// Sets the frequency penalty.
    #[must_use]
    pub fn frequency_penalty(mut self, penalty: f32) -> Self {
        self.params.frequency_penalty = Some(penalty);
        self
    }

    /// Sets the stop sequences.
    #[must_use]
    pub fn stop_sequences(mut self, sequences: Vec<String>) -> Self {
        self.params.stop_sequences = sequences;
        self
    }

    /// Builds the params.
    #[must_use]
    pub fn build(self) -> ParticipantParams {
        self.params
    }
}

/// Configuration for a dialogue participant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DialogueParticipant {
    /// Unique identifier.
    pub id: ParticipantId,
    /// Provider ID (e.g., "openai").
    pub provider_id: String,
    /// Model ID (e.g., "gpt-4").
    pub model_id: String,
    /// Display name (e.g., "gpt-4" or custom).
    pub display_name: String,
    /// System prompt for this participant.
    pub system_prompt: Option<String>,
    /// LLM parameters.
    pub params: ParticipantParams,
    /// Display color.
    pub color: ParticipantColor,
    /// Whether the participant is active.
    pub active: bool,
}

impl DialogueParticipant {
    /// Creates a new participant from provider:model string.
    ///
    /// # Arguments
    /// * `provider_model` - String in format "provider:model" (e.g., "openai:gpt-4")
    /// * `index` - Index for automatic color assignment
    #[must_use]
    pub fn from_provider_model(provider_model: &str, index: usize) -> Option<Self> {
        let parts: Vec<&str> = provider_model.split(':').collect();
        if parts.len() != 2 {
            return None;
        }

        let provider_id = parts[0].to_string();
        let model_id = parts[1].to_string();
        let display_name = model_id.clone();

        Some(Self {
            id: ParticipantId::new(),
            provider_id,
            model_id,
            display_name,
            system_prompt: None,
            params: ParticipantParams::default(),
            color: ParticipantColor::for_index(index),
            active: true,
        })
    }

    /// Creates a new builder for a participant.
    #[must_use]
    pub fn builder(
        provider_id: impl Into<String>,
        model_id: impl Into<String>,
    ) -> ParticipantBuilder {
        ParticipantBuilder::new(provider_id, model_id)
    }

    /// Returns the full provider:model identifier.
    #[must_use]
    pub fn provider_model(&self) -> String {
        format!("{}:{}", self.provider_id, self.model_id)
    }
}

/// Builder for dialogue participants.
#[derive(Debug)]
pub struct ParticipantBuilder {
    participant: DialogueParticipant,
}

impl ParticipantBuilder {
    /// Creates a new builder.
    #[must_use]
    pub fn new(provider_id: impl Into<String>, model_id: impl Into<String>) -> Self {
        let model = model_id.into();
        Self {
            participant: DialogueParticipant {
                id: ParticipantId::new(),
                provider_id: provider_id.into(),
                model_id: model.clone(),
                display_name: model,
                system_prompt: None,
                params: ParticipantParams::default(),
                color: ParticipantColor::Orange,
                active: true,
            },
        }
    }

    /// Sets the display name.
    #[must_use]
    pub fn display_name(mut self, name: impl Into<String>) -> Self {
        self.participant.display_name = name.into();
        self
    }

    /// Sets the system prompt.
    #[must_use]
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.participant.system_prompt = Some(prompt.into());
        self
    }

    /// Sets the parameters.
    #[must_use]
    pub fn params(mut self, params: ParticipantParams) -> Self {
        self.participant.params = params;
        self
    }

    /// Sets the color.
    #[must_use]
    pub fn color(mut self, color: ParticipantColor) -> Self {
        self.participant.color = color;
        self
    }

    /// Sets the color by index.
    #[must_use]
    pub fn color_index(mut self, index: usize) -> Self {
        self.participant.color = ParticipantColor::for_index(index);
        self
    }

    /// Builds the participant.
    #[must_use]
    pub fn build(self) -> DialogueParticipant {
        self.participant
    }
}

/// An active participant with a live provider connection.
pub struct ActiveParticipant {
    /// Participant configuration.
    pub config: DialogueParticipant,
    /// Live provider instance.
    pub provider: Arc<dyn LLMProvider>,
}

impl std::fmt::Debug for ActiveParticipant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ActiveParticipant")
            .field("config", &self.config)
            .field("provider", &"<dyn LLMProvider>")
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_provider_model_parses_valid_format() {
        let participant = DialogueParticipant::from_provider_model("openai:gpt-4", 0);
        assert!(participant.is_some());
        let p = participant.unwrap();
        assert_eq!(p.provider_id, "openai");
        assert_eq!(p.model_id, "gpt-4");
        assert_eq!(p.display_name, "gpt-4");
        assert!(p.active);
    }

    #[test]
    fn from_provider_model_rejects_invalid_format() {
        assert!(DialogueParticipant::from_provider_model("invalid", 0).is_none());
        assert!(DialogueParticipant::from_provider_model("too:many:colons", 0).is_none());
        assert!(DialogueParticipant::from_provider_model("", 0).is_none());
    }

    #[test]
    fn color_cycles_correctly() {
        assert_eq!(ParticipantColor::for_index(0), ParticipantColor::Orange);
        assert_eq!(ParticipantColor::for_index(1), ParticipantColor::Cyan);
        assert_eq!(ParticipantColor::for_index(8), ParticipantColor::Orange); // wraps around
    }

    #[test]
    fn color_rgb_returns_valid_values() {
        let (r, g, b) = ParticipantColor::Orange.rgb();
        assert!(r > 0 && g > 0 && b > 0);
    }
}
