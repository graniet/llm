//! Dialogue controller methods for multi-LLM conversations.

use crate::dialogue::{
    DialogueConfig, DialogueController, DialogueMode, DialogueParticipant, StopReason, TurnMode,
};
use crate::provider::{ProviderFactory, ProviderId};
use crate::runtime::overlay::{DialogueBuilderResult, DialogueBuilderState};
use crate::runtime::{AppStatus, OverlayState};

use super::AppController;

impl AppController {
    /// Opens the dialogue builder overlay.
    pub fn open_dialogue_builder(&mut self) -> bool {
        self.state.overlay = OverlayState::DialogueBuilder(DialogueBuilderState::new());
        true
    }

    /// Starts a dialogue with the specified participants.
    pub fn start_dialogue(&mut self, participants: Vec<&str>) -> bool {
        if participants.len() < 2 {
            self.set_status(AppStatus::Error(
                "Multi-LLM dialogue requires at least 2 participants".to_string(),
            ));
            return false;
        }

        let factory = ProviderFactory::new(&self.state.config, &self.state.provider_registry);

        let mut dialogue_participants = Vec::new();
        let mut participant_providers = Vec::new();

        for (index, provider_model) in participants.iter().enumerate() {
            let Some(participant) = DialogueParticipant::from_provider_model(provider_model, index)
            else {
                self.set_status(AppStatus::Error(format!(
                    "Invalid participant format: {}. Use provider:model",
                    provider_model
                )));
                return false;
            };

            // Build provider for this participant
            let provider_id = ProviderId::new(&participant.provider_id);
            let selection = crate::provider::ProviderSelection {
                provider_id,
                model: Some(participant.model_id.clone()),
            };
            let overrides = crate::provider::ProviderOverrides::default();

            match factory.build(&selection, overrides) {
                Ok(instance) => {
                    participant_providers.push((participant.clone(), instance.provider));
                    dialogue_participants.push(participant);
                }
                Err(_) => {
                    self.set_status(AppStatus::Error(format!(
                        "Failed to create provider for {}",
                        provider_model
                    )));
                    return false;
                }
            }
        }

        // Initialize dialogue in conversation state
        let config = DialogueConfig {
            mode: DialogueMode::FreeForm,
            initial_prompt: String::new(),
            turn_mode: crate::dialogue::TurnMode::RoundRobin,
        };

        if let Some(conv) = self.state.conversations.active_mut() {
            conv.start_dialogue(
                DialogueMode::FreeForm,
                config.clone(),
                dialogue_participants.clone(),
            );
        }

        // Initialize the dialogue controller with participants
        let mut controller = DialogueController::new(config);
        for (participant, provider) in participant_providers {
            controller.add_participant(participant, provider);
        }
        self.state.dialogue_controller = Some(controller);

        self.push_notice(format!(
            "Dialogue started with {} participants. Type /continue to begin or send a message.",
            dialogue_participants.len()
        ));
        self.set_status(AppStatus::Idle);
        true
    }

    /// Invites a new participant to the active dialogue.
    pub fn invite_dialogue_participant(&mut self, provider_model: &str) -> bool {
        let Some(ref mut controller) = self.state.dialogue_controller else {
            self.set_status(AppStatus::Error("No active dialogue".to_string()));
            return false;
        };

        let index = controller.participants().len();
        let Some(participant) = DialogueParticipant::from_provider_model(provider_model, index)
        else {
            self.set_status(AppStatus::Error(format!(
                "Invalid participant format: {}. Use provider:model",
                provider_model
            )));
            return false;
        };

        // Verify provider exists
        let provider_id = ProviderId::new(&participant.provider_id);
        if self.state.provider_registry.get(&provider_id).is_none() {
            self.set_status(AppStatus::Error(format!(
                "Unknown provider: {}",
                participant.provider_id
            )));
            return false;
        }

        // Build provider for the participant
        let factory = ProviderFactory::new(&self.state.config, &self.state.provider_registry);
        let selection = crate::provider::ProviderSelection {
            provider_id,
            model: Some(participant.model_id.clone()),
        };
        let overrides = crate::provider::ProviderOverrides::default();

        let Ok(instance) = factory.build(&selection, overrides) else {
            self.set_status(AppStatus::Error(format!(
                "Failed to create provider for {}",
                provider_model
            )));
            return false;
        };

        controller.add_participant(participant.clone(), instance.provider);

        // Also add to conversation state
        if let Some(conv) = self.state.conversations.active_mut() {
            conv.participants.push(participant.clone());
        }

        self.push_notice(format!("Invited {} to dialogue", participant.display_name));
        true
    }

    /// Kicks a participant from the active dialogue.
    pub fn kick_dialogue_participant(&mut self, name: &str) -> bool {
        let Some(ref mut controller) = self.state.dialogue_controller else {
            self.set_status(AppStatus::Error("No active dialogue".to_string()));
            return false;
        };

        let Some(participant) = controller.find_participant_by_name(name) else {
            self.set_status(AppStatus::Error(format!(
                "Participant '{}' not found",
                name
            )));
            return false;
        };

        let id = participant.config.id;
        let display_name = participant.config.display_name.clone();

        if controller.kick_participant(id) {
            // Also update conversation state
            if let Some(conv) = self.state.conversations.active_mut() {
                if let Some(pos) = conv.participants.iter().position(|p| p.id == id) {
                    conv.participants[pos].active = false;
                }
            }

            self.push_notice(format!("Kicked {} from dialogue", display_name));
            true
        } else {
            self.set_status(AppStatus::Error(format!("Failed to kick {}", display_name)));
            false
        }
    }

    /// Stops the active dialogue.
    pub fn stop_dialogue(&mut self) -> bool {
        let Some(ref mut controller) = self.state.dialogue_controller else {
            self.set_status(AppStatus::Error("No active dialogue".to_string()));
            return false;
        };

        controller.stop(StopReason::UserRequested);

        // Clear dialogue state from conversation
        if let Some(conv) = self.state.conversations.active_mut() {
            conv.end_dialogue();
        }

        self.state.dialogue_controller = None;
        self.push_notice("Dialogue stopped".to_string());
        self.set_status(AppStatus::Idle);
        true
    }

    /// Starts a dialogue from the builder result.
    pub fn start_dialogue_from_builder(&mut self, result: DialogueBuilderResult) -> bool {
        if result.participants.len() < 2 {
            self.set_status(AppStatus::Error(
                "Multi-LLM dialogue requires at least 2 participants".to_string(),
            ));
            return false;
        }

        let factory = ProviderFactory::new(&self.state.config, &self.state.provider_registry);

        // Convert draft participants to DialogueParticipant and build providers
        let mut dialogue_participants: Vec<DialogueParticipant> = Vec::new();
        let mut participant_providers = Vec::new();

        for draft in &result.participants {
            let participant = DialogueParticipant {
                id: crate::dialogue::ParticipantId::new(),
                provider_id: draft.provider_id.clone(),
                model_id: draft.model_id.clone(),
                display_name: draft.display_name.clone(),
                system_prompt: draft.system_prompt.clone(),
                params: crate::dialogue::ParticipantParams::default(),
                color: draft.color,
                active: true,
            };

            // Build provider for this participant
            let provider_id = ProviderId::new(&draft.provider_id);
            let selection = crate::provider::ProviderSelection {
                provider_id,
                model: Some(draft.model_id.clone()),
            };
            let overrides = crate::provider::ProviderOverrides::default();

            match factory.build(&selection, overrides) {
                Ok(instance) => {
                    participant_providers.push((participant.clone(), instance.provider));
                    dialogue_participants.push(participant);
                }
                Err(_) => {
                    self.set_status(AppStatus::Error(format!(
                        "Failed to create provider for {}:{}",
                        draft.provider_id, draft.model_id
                    )));
                    return false;
                }
            }
        }

        // Initialize dialogue in conversation state
        let config = DialogueConfig {
            mode: result.mode,
            initial_prompt: result.initial_prompt.clone(),
            turn_mode: TurnMode::RoundRobin,
        };

        if let Some(conv) = self.state.conversations.active_mut() {
            conv.start_dialogue(result.mode, config.clone(), dialogue_participants.clone());
        }

        // Initialize the dialogue controller with participants
        let mut controller = DialogueController::new(config);
        for (participant, provider) in participant_providers {
            controller.add_participant(participant, provider);
        }
        self.state.dialogue_controller = Some(controller);

        self.push_notice(format!(
            "Dialogue started with {} participants. Type /continue to begin or send a message.",
            dialogue_participants.len()
        ));
        self.set_status(AppStatus::Idle);
        true
    }

    /// Returns whether a dialogue is currently active.
    pub fn is_dialogue_active(&self) -> bool {
        self.state.dialogue_controller.is_some()
    }

    /// Gets the active dialogue controller.
    #[allow(dead_code)]
    pub fn dialogue_controller(&self) -> Option<&DialogueController> {
        self.state.dialogue_controller.as_ref()
    }

    /// Gets a mutable reference to the active dialogue controller.
    #[allow(dead_code)]
    pub fn dialogue_controller_mut(&mut self) -> Option<&mut DialogueController> {
        self.state.dialogue_controller.as_mut()
    }

    /// Continues the dialogue with the next participant's turn.
    pub fn continue_dialogue(&mut self) -> bool {
        let Some(ref dialogue) = self.state.dialogue_controller else {
            self.set_status(AppStatus::Error("No active dialogue".to_string()));
            return false;
        };

        let Some(next_participant) = dialogue.next_participant() else {
            self.set_status(AppStatus::Error("No active participants".to_string()));
            return false;
        };

        let participant_name = next_participant.config.display_name.clone();
        let participant_color = next_participant.config.color;
        let participant_system_prompt = next_participant.config.system_prompt.clone();
        let provider_id = ProviderId::new(&next_participant.config.provider_id);

        // Build provider for the next participant with their system prompt
        let factory = ProviderFactory::new(&self.state.config, &self.state.provider_registry);
        let selection = crate::provider::ProviderSelection {
            provider_id,
            model: Some(next_participant.config.model_id.clone()),
        };
        let overrides = crate::provider::ProviderOverrides {
            system: participant_system_prompt,
            ..Default::default()
        };

        let Ok(instance) = factory.build(&selection, overrides) else {
            self.set_status(AppStatus::Error(format!(
                "Failed to create provider for {}",
                participant_name
            )));
            return false;
        };

        // Create a placeholder message for the next participant
        let Some(conv_id) = self.state.active_conversation_id() else {
            return false;
        };

        // Add a placeholder assistant message with participant metadata
        let assistant_id =
            self.append_dialogue_placeholder(conv_id, &participant_name, participant_color);

        // Build context messages from dialogue history
        let Some(conversation) = self.state.active_conversation().cloned() else {
            return false;
        };

        // Get messages for context, passing current participant name
        let messages = self.build_dialogue_messages(&conversation, &participant_name);

        // Start stream for this participant
        let request = crate::runtime::streaming::StreamRequest {
            conversation_id: conv_id,
            message_id: assistant_id,
            provider: instance.provider,
            messages,
            capabilities: instance.capabilities,
        };

        self.stream_manager.start(request);
        self.set_status(AppStatus::Streaming);
        self.push_notice(format!("{} is responding...", participant_name));
        true
    }

    /// Appends a placeholder message for a dialogue participant.
    fn append_dialogue_placeholder(
        &mut self,
        conv_id: crate::conversation::ConversationId,
        participant_name: &str,
        color: crate::dialogue::ParticipantColor,
    ) -> crate::conversation::MessageId {
        use crate::conversation::{ConversationMessage, MessageKind, MessageRole, MessageState};

        let mut message =
            ConversationMessage::new(MessageRole::Assistant, MessageKind::Text(String::new()));
        message.state = MessageState::Streaming;
        message.metadata.participant_name = Some(participant_name.to_string());
        message.metadata.participant_color = Some(color);

        let id = message.id;
        if let Some(conv) = self.state.conversations.active_mut() {
            if conv.id == conv_id {
                conv.push_message(message);
            }
        }
        id
    }

    /// Builds chat messages from dialogue history for context.
    ///
    /// For dialogue mode, messages from OTHER participants are sent as User role
    /// so the current participant sees them as input to respond to.
    fn build_dialogue_messages(
        &self,
        conversation: &crate::conversation::Conversation,
        current_participant: &str,
    ) -> Vec<llm::chat::ChatMessage> {
        use crate::conversation::{MessageKind, MessageRole};
        use llm::chat::ChatMessage;

        let mut messages = Vec::new();

        // Add dialogue initial prompt as user message if present
        if let Some(ref config) = conversation.dialogue_config {
            if !config.initial_prompt.is_empty() {
                messages.push(
                    ChatMessage::user()
                        .content(config.initial_prompt.clone())
                        .build(),
                );
            }
        }

        // Add conversation messages
        for msg in &conversation.messages {
            if let MessageKind::Text(ref content) = msg.kind {
                if content.is_empty() {
                    continue;
                }

                // Determine the role based on who sent the message
                let role = match msg.role {
                    MessageRole::User => llm::chat::ChatRole::User,
                    MessageRole::Assistant => {
                        // Check if this message is from the current participant
                        if let Some(ref name) = msg.metadata.participant_name {
                            if name == current_participant {
                                // This participant's own previous messages = Assistant
                                llm::chat::ChatRole::Assistant
                            } else {
                                // Other participants' messages = User (so we respond to them)
                                llm::chat::ChatRole::User
                            }
                        } else {
                            // No participant name (shouldn't happen in dialogue)
                            llm::chat::ChatRole::Assistant
                        }
                    }
                    MessageRole::Tool | MessageRole::Error => continue,
                };

                // Add participant prefix for clarity
                let content = if let Some(ref name) = msg.metadata.participant_name {
                    format!("[{}] {}", name, content)
                } else {
                    content.clone()
                };

                messages.push(ChatMessage {
                    role,
                    message_type: llm::chat::MessageType::Text,
                    content,
                });
            }
        }

        messages
    }
}
