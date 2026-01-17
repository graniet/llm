//! Dialogue controller for managing multi-LLM conversations.

use std::collections::VecDeque;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use llm::chat::{ChatMessage, ChatRole, Usage};
use llm::LLMProvider;

use super::events::{DialogueEvent, StopReason};
use super::participant::{ActiveParticipant, DialogueParticipant, ParticipantId};
use super::DialogueConfig;

/// Turn mode for participant ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TurnMode {
    /// Strict alternation between participants.
    #[default]
    RoundRobin,
    /// The current speaker designates the next (future).
    Directed,
    /// Any participant can speak (when user intervenes).
    Free,
}

/// State of the dialogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DialogueState {
    /// Dialogue not yet started.
    #[default]
    Idle,
    /// A participant is currently streaming a response.
    Streaming,
    /// Waiting for the next turn.
    WaitingNext,
    /// Dialogue is paused.
    Paused,
    /// Dialogue has stopped.
    Stopped,
}

/// A recorded turn in the dialogue history.
#[derive(Debug, Clone)]
pub struct DialogueTurn {
    /// ID of the participant who spoke.
    pub participant_id: ParticipantId,
    /// Content of the turn.
    pub content: String,
    /// When the turn occurred.
    pub timestamp: DateTime<Utc>,
    /// Token usage for this turn.
    pub usage: Option<Usage>,
}

/// Controller for managing a multi-LLM dialogue.
pub struct DialogueController {
    /// Active participants with live providers.
    participants: Vec<ActiveParticipant>,
    /// Configuration for the dialogue.
    config: DialogueConfig,
    /// Current participant index (for round-robin).
    current_index: usize,
    /// Current state of the dialogue.
    state: DialogueState,
    /// Queue of user messages to inject.
    user_message_queue: VecDeque<String>,
    /// History of turns.
    history: Vec<DialogueTurn>,
    /// Event sender for notifying listeners.
    event_sender: Option<mpsc::UnboundedSender<DialogueEvent>>,
}

impl DialogueController {
    /// Creates a new dialogue controller.
    #[must_use]
    pub fn new(config: DialogueConfig) -> Self {
        Self {
            participants: Vec::new(),
            config,
            current_index: 0,
            state: DialogueState::Idle,
            user_message_queue: VecDeque::new(),
            history: Vec::new(),
            event_sender: None,
        }
    }

    /// Sets the event sender for receiving dialogue events.
    pub fn set_event_sender(&mut self, sender: mpsc::UnboundedSender<DialogueEvent>) {
        self.event_sender = Some(sender);
    }

    /// Creates an event receiver channel.
    #[must_use]
    pub fn create_event_channel(&mut self) -> mpsc::UnboundedReceiver<DialogueEvent> {
        let (tx, rx) = mpsc::unbounded_channel();
        self.event_sender = Some(tx);
        rx
    }

    /// Returns the current state.
    #[must_use]
    pub const fn state(&self) -> DialogueState {
        self.state
    }

    /// Returns the configuration.
    #[must_use]
    pub const fn config(&self) -> &DialogueConfig {
        &self.config
    }

    /// Returns the participants.
    #[must_use]
    pub fn participants(&self) -> &[ActiveParticipant] {
        &self.participants
    }

    /// Returns the dialogue history.
    #[must_use]
    pub fn history(&self) -> &[DialogueTurn] {
        &self.history
    }

    /// Returns the number of active participants.
    #[must_use]
    pub fn active_participant_count(&self) -> usize {
        self.participants.iter().filter(|p| p.config.active).count()
    }

    /// Adds a participant to the dialogue.
    pub fn add_participant(
        &mut self,
        config: DialogueParticipant,
        provider: Arc<dyn LLMProvider>,
    ) -> ParticipantId {
        let id = config.id;
        let name = config.display_name.clone();

        self.participants
            .push(ActiveParticipant { config, provider });

        self.emit_event(DialogueEvent::ParticipantJoined {
            participant_id: id,
            participant_name: name,
        });

        id
    }

    /// Removes a participant from the dialogue.
    pub fn remove_participant(&mut self, id: ParticipantId) -> bool {
        if let Some(pos) = self.participants.iter().position(|p| p.config.id == id) {
            self.participants.remove(pos);
            self.emit_event(DialogueEvent::ParticipantLeft { participant_id: id });

            // Adjust current index if needed
            if self.current_index >= self.participants.len() && !self.participants.is_empty() {
                self.current_index = 0;
            }

            // Check if dialogue should stop
            if self.active_participant_count() == 0 {
                self.stop(StopReason::AllParticipantsLeft);
            }

            true
        } else {
            false
        }
    }

    /// Deactivates a participant (kick without removing history).
    pub fn kick_participant(&mut self, id: ParticipantId) -> bool {
        if let Some(participant) = self.participants.iter_mut().find(|p| p.config.id == id) {
            participant.config.active = false;
            self.emit_event(DialogueEvent::ParticipantLeft { participant_id: id });

            if self.active_participant_count() == 0 {
                self.stop(StopReason::AllParticipantsLeft);
            }

            true
        } else {
            false
        }
    }

    /// Reactivates a participant (invite back).
    pub fn invite_participant(&mut self, id: ParticipantId) -> bool {
        if let Some(participant) = self.participants.iter_mut().find(|p| p.config.id == id) {
            participant.config.active = true;
            let name = participant.config.display_name.clone();
            self.emit_event(DialogueEvent::ParticipantJoined {
                participant_id: id,
                participant_name: name,
            });
            true
        } else {
            false
        }
    }

    /// Queues a user message to be injected into the dialogue.
    pub fn inject_user_message(&mut self, content: String) {
        self.user_message_queue.push_back(content.clone());
        self.emit_event(DialogueEvent::UserMessage { content });
    }

    /// Starts the dialogue with the initial prompt.
    pub fn start(&mut self) {
        if self.participants.is_empty() {
            return;
        }

        self.state = DialogueState::WaitingNext;
        self.emit_event(DialogueEvent::Started);

        // Queue the initial prompt as a user message if not empty
        if !self.config.initial_prompt.is_empty() {
            self.user_message_queue
                .push_back(self.config.initial_prompt.clone());
        }
    }

    /// Stops the dialogue.
    pub fn stop(&mut self, reason: StopReason) {
        self.state = DialogueState::Stopped;
        self.emit_event(DialogueEvent::Stopped { reason });
    }

    /// Pauses the dialogue.
    pub fn pause(&mut self) {
        if self.state == DialogueState::WaitingNext {
            self.state = DialogueState::Paused;
        }
    }

    /// Resumes a paused dialogue.
    pub fn resume(&mut self) {
        if self.state == DialogueState::Paused {
            self.state = DialogueState::WaitingNext;
        }
    }

    /// Returns the next participant for a turn.
    #[must_use]
    pub fn next_participant(&self) -> Option<&ActiveParticipant> {
        if self.participants.is_empty() {
            return None;
        }

        // Find next active participant
        let start = self.current_index;
        let mut idx = start;
        loop {
            if self.participants[idx].config.active {
                return Some(&self.participants[idx]);
            }
            idx = (idx + 1) % self.participants.len();
            if idx == start {
                return None; // No active participants
            }
        }
    }

    /// Advances to the next turn.
    pub fn advance_turn(&mut self) {
        if self.participants.is_empty() {
            return;
        }

        // Move to next participant
        self.current_index = (self.current_index + 1) % self.participants.len();

        // Skip inactive participants
        let start = self.current_index;
        while !self.participants[self.current_index].config.active {
            self.current_index = (self.current_index + 1) % self.participants.len();
            if self.current_index == start {
                break;
            }
        }
    }

    /// Records a completed turn.
    pub fn record_turn(
        &mut self,
        participant_id: ParticipantId,
        content: String,
        usage: Option<Usage>,
    ) {
        let turn = DialogueTurn {
            participant_id,
            content: content.clone(),
            timestamp: Utc::now(),
            usage: usage.clone(),
        };
        self.history.push(turn);

        self.emit_event(DialogueEvent::TurnCompleted {
            participant_id,
            content,
            usage,
        });
    }

    /// Drains any queued user messages.
    pub fn drain_user_messages(&mut self) -> Vec<String> {
        self.user_message_queue.drain(..).collect()
    }

    /// Builds the message context for a participant's turn.
    #[must_use]
    pub fn build_context_messages(&self, participant: &ActiveParticipant) -> Vec<ChatMessage> {
        let mut messages = Vec::new();

        // Add dialogue history
        for turn in &self.history {
            // Determine the role based on who spoke
            let role = if turn.participant_id == participant.config.id {
                ChatRole::Assistant
            } else {
                ChatRole::User
            };

            // Add prefix to identify speaker
            let speaker_name = self
                .participants
                .iter()
                .find(|p| p.config.id == turn.participant_id)
                .map(|p| p.config.display_name.as_str())
                .unwrap_or("Unknown");

            let content = format!("[{}] {}", speaker_name, turn.content);
            messages.push(ChatMessage {
                role,
                message_type: llm::chat::MessageType::Text,
                content,
            });
        }

        // Add any queued user messages
        for msg in &self.user_message_queue {
            messages.push(ChatMessage::user().content(msg.clone()).build());
        }

        messages
    }

    /// Emits an event to listeners.
    fn emit_event(&self, event: DialogueEvent) {
        if let Some(sender) = &self.event_sender {
            let _ = sender.send(event);
        }
    }

    /// Sets the dialogue state.
    pub fn set_state(&mut self, state: DialogueState) {
        self.state = state;
    }

    /// Gets participant by ID.
    #[must_use]
    pub fn get_participant(&self, id: ParticipantId) -> Option<&ActiveParticipant> {
        self.participants.iter().find(|p| p.config.id == id)
    }

    /// Gets participant by index.
    #[must_use]
    pub fn get_participant_by_index(&self, index: usize) -> Option<&ActiveParticipant> {
        self.participants.get(index)
    }

    /// Finds a participant by display name.
    #[must_use]
    pub fn find_participant_by_name(&self, name: &str) -> Option<&ActiveParticipant> {
        self.participants
            .iter()
            .find(|p| p.config.display_name.eq_ignore_ascii_case(name))
    }
}

impl std::fmt::Debug for DialogueController {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DialogueController")
            .field("participants", &self.participants.len())
            .field("state", &self.state)
            .field("current_index", &self.current_index)
            .field("history_len", &self.history.len())
            .finish()
    }
}
