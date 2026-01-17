//! Dialogue events for tracking dialogue state changes.

use llm::chat::Usage;

use super::ParticipantId;

/// Events emitted during a dialogue session.
#[derive(Debug, Clone)]
pub enum DialogueEvent {
    /// The dialogue has started.
    Started,

    /// A new turn is starting.
    TurnStarted {
        /// ID of the participant whose turn is starting.
        participant_id: ParticipantId,
        /// Display name of the participant.
        participant_name: String,
    },

    /// A token was received during streaming.
    Token {
        /// ID of the participant generating the token.
        participant_id: ParticipantId,
        /// The token content.
        content: String,
    },

    /// A turn has completed.
    TurnCompleted {
        /// ID of the participant who completed.
        participant_id: ParticipantId,
        /// Full content of the turn.
        content: String,
        /// Token usage if available.
        usage: Option<Usage>,
    },

    /// A user message was injected into the dialogue.
    UserMessage {
        /// Content of the user message.
        content: String,
    },

    /// A new participant joined the dialogue.
    ParticipantJoined {
        /// ID of the new participant.
        participant_id: ParticipantId,
        /// Display name of the participant.
        participant_name: String,
    },

    /// A participant left the dialogue.
    ParticipantLeft {
        /// ID of the participant who left.
        participant_id: ParticipantId,
    },

    /// The dialogue has stopped.
    Stopped {
        /// Reason for stopping.
        reason: StopReason,
    },

    /// An error occurred.
    Error {
        /// Error message.
        message: String,
        /// Whether the dialogue can continue.
        recoverable: bool,
    },
}

/// Reason for dialogue stopping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StopReason {
    /// User requested stop via /stop command.
    UserRequested,
    /// An unrecoverable error occurred.
    Error(String),
    /// All participants have left.
    AllParticipantsLeft,
    /// Dialogue completed naturally (for future use).
    Completed,
}

impl std::fmt::Display for StopReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UserRequested => write!(f, "User requested stop"),
            Self::Error(msg) => write!(f, "Error: {msg}"),
            Self::AllParticipantsLeft => write!(f, "All participants left"),
            Self::Completed => write!(f, "Dialogue completed"),
        }
    }
}
