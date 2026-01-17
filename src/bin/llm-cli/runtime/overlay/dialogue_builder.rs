//! Dialogue builder overlay for configuring multi-LLM dialogues.

use crate::dialogue::{DialogueMode, ParticipantColor};

/// Step in the dialogue builder wizard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DialogueBuilderStep {
    /// Selecting participants.
    #[default]
    Participants,
    /// Configuring a specific participant.
    ConfigureParticipant,
    /// Configuring dialogue mode.
    Mode,
    /// Setting initial prompt.
    InitialPrompt,
    /// Review and confirm.
    Review,
}

/// Field being edited for a participant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ParticipantField {
    /// Display name.
    #[default]
    DisplayName,
    /// System prompt.
    SystemPrompt,
}

impl DialogueBuilderStep {
    /// Returns the next step.
    pub fn next(self) -> Option<Self> {
        match self {
            Self::Participants => Some(Self::Mode),
            Self::ConfigureParticipant => Some(Self::Participants),
            Self::Mode => Some(Self::InitialPrompt),
            Self::InitialPrompt => Some(Self::Review),
            Self::Review => None,
        }
    }

    /// Returns the previous step.
    pub fn prev(self) -> Option<Self> {
        match self {
            Self::Participants => None,
            Self::ConfigureParticipant => Some(Self::Participants),
            Self::Mode => Some(Self::Participants),
            Self::InitialPrompt => Some(Self::Mode),
            Self::Review => Some(Self::InitialPrompt),
        }
    }

    /// Returns the display name for the step.
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::Participants => "Participants",
            Self::ConfigureParticipant => "Configure",
            Self::Mode => "Mode",
            Self::InitialPrompt => "Initial Prompt",
            Self::Review => "Review",
        }
    }
}

impl ParticipantField {
    /// Returns the next field.
    pub fn next(self) -> Self {
        match self {
            Self::DisplayName => Self::SystemPrompt,
            Self::SystemPrompt => Self::DisplayName,
        }
    }

    /// Returns the display name for the field.
    pub const fn display_name(self) -> &'static str {
        match self {
            Self::DisplayName => "Display Name",
            Self::SystemPrompt => "System Prompt",
        }
    }
}

/// A draft participant being configured.
#[derive(Debug, Clone)]
pub struct ParticipantDraft {
    /// Provider ID (e.g., "openai").
    pub provider_id: String,
    /// Model ID (e.g., "gpt-4").
    pub model_id: String,
    /// Display name.
    pub display_name: String,
    /// System prompt (optional).
    pub system_prompt: Option<String>,
    /// Assigned color.
    pub color: ParticipantColor,
}

impl ParticipantDraft {
    /// Creates a new draft from provider:model string.
    pub fn from_provider_model(provider_model: &str, index: usize) -> Option<Self> {
        let parts: Vec<&str> = provider_model.split(':').collect();
        if parts.len() != 2 {
            return None;
        }

        let provider_id = parts[0].to_string();
        let model_id = parts[1].to_string();
        let display_name = model_id.clone();

        Some(Self {
            provider_id,
            model_id,
            display_name,
            system_prompt: None,
            color: ParticipantColor::for_index(index),
        })
    }
}

/// State for the dialogue builder overlay.
#[derive(Debug, Clone)]
pub struct DialogueBuilderState {
    /// Current step in the wizard.
    pub step: DialogueBuilderStep,
    /// Draft participants.
    pub participants: Vec<ParticipantDraft>,
    /// Selected dialogue mode.
    pub mode: DialogueMode,
    /// Initial prompt for the dialogue.
    pub initial_prompt: String,
    /// Current input buffer (for text entry).
    pub input: String,
    /// Error message to display.
    pub error: Option<String>,
    /// Currently selected item in lists.
    pub selected: usize,
    /// Available modes.
    pub available_modes: Vec<DialogueMode>,
    /// Index of participant being configured (for ConfigureParticipant step).
    pub editing_participant: usize,
    /// Field being edited for the participant.
    pub editing_field: ParticipantField,
}

impl Default for DialogueBuilderState {
    fn default() -> Self {
        Self::new()
    }
}

impl DialogueBuilderState {
    /// Creates a new dialogue builder state.
    pub fn new() -> Self {
        Self {
            step: DialogueBuilderStep::default(),
            participants: Vec::new(),
            mode: DialogueMode::FreeForm,
            initial_prompt: String::new(),
            input: String::new(),
            error: None,
            selected: 0,
            available_modes: vec![
                DialogueMode::FreeForm,
                DialogueMode::Collaboration,
                DialogueMode::Debate,
                DialogueMode::Critique,
                DialogueMode::Custom,
            ],
            editing_participant: 0,
            editing_field: ParticipantField::default(),
        }
    }

    /// Advances to the next step if possible.
    pub fn next_step(&mut self) -> bool {
        if let Some(next) = self.step.next() {
            self.step = next;
            self.selected = 0;
            self.error = None;
            true
        } else {
            false
        }
    }

    /// Goes back to the previous step if possible.
    pub fn prev_step(&mut self) -> bool {
        if let Some(prev) = self.step.prev() {
            self.step = prev;
            self.selected = 0;
            self.error = None;
            true
        } else {
            false
        }
    }

    /// Adds a participant from the current input.
    pub fn add_participant(&mut self) -> bool {
        let input = self.input.trim();
        if input.is_empty() {
            return false;
        }

        // Try to parse as provider:model
        let provider_model = input.trim_start_matches('@');
        let index = self.participants.len();

        if let Some(draft) = ParticipantDraft::from_provider_model(provider_model, index) {
            self.participants.push(draft);
            self.input.clear();
            self.error = None;
            true
        } else {
            self.error = Some("Invalid format. Use provider:model".to_string());
            false
        }
    }

    /// Removes the selected participant.
    pub fn remove_participant(&mut self) {
        if !self.participants.is_empty() && self.selected < self.participants.len() {
            self.participants.remove(self.selected);
            if self.selected > 0 && self.selected >= self.participants.len() {
                self.selected = self.participants.len().saturating_sub(1);
            }
        }
    }

    /// Pushes a character to the input.
    pub fn push_input(&mut self, ch: char) {
        self.input.push(ch);
    }

    /// Pops a character from the input.
    pub fn pop_input(&mut self) {
        self.input.pop();
    }

    /// Selects the next item.
    pub fn next(&mut self) {
        let max = match self.step {
            DialogueBuilderStep::Participants => self.participants.len(),
            DialogueBuilderStep::Mode => self.available_modes.len(),
            _ => 0,
        };
        if max > 0 {
            self.selected = (self.selected + 1).min(max.saturating_sub(1));
        }
    }

    /// Selects the previous item.
    pub fn prev(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
        }
    }

    /// Selects the current mode (when on Mode step).
    pub fn select_mode(&mut self) {
        if self.step == DialogueBuilderStep::Mode && self.selected < self.available_modes.len() {
            self.mode = self.available_modes[self.selected];
        }
    }

    /// Enters configuration mode for the selected participant.
    pub fn configure_participant(&mut self) -> bool {
        if self.step == DialogueBuilderStep::Participants
            && !self.participants.is_empty()
            && self.selected < self.participants.len()
        {
            self.editing_participant = self.selected;
            self.editing_field = ParticipantField::DisplayName;
            // Load current display name into input
            self.input = self.participants[self.editing_participant]
                .display_name
                .clone();
            self.step = DialogueBuilderStep::ConfigureParticipant;
            self.error = None;
            true
        } else {
            false
        }
    }

    /// Exits configuration mode, saving changes.
    pub fn finish_configure(&mut self) {
        if self.step == DialogueBuilderStep::ConfigureParticipant {
            // Save current field
            self.save_current_field();
            self.step = DialogueBuilderStep::Participants;
            self.input.clear();
            self.error = None;
        }
    }

    /// Switches to editing the next field.
    pub fn next_field(&mut self) {
        if self.step == DialogueBuilderStep::ConfigureParticipant {
            // Save current field first
            self.save_current_field();
            // Switch to next field
            self.editing_field = self.editing_field.next();
            // Load the new field's value
            self.load_current_field();
        }
    }

    /// Saves the current input to the appropriate field.
    fn save_current_field(&mut self) {
        if self.editing_participant < self.participants.len() {
            let participant = &mut self.participants[self.editing_participant];
            match self.editing_field {
                ParticipantField::DisplayName => {
                    if !self.input.trim().is_empty() {
                        participant.display_name = self.input.trim().to_string();
                    }
                }
                ParticipantField::SystemPrompt => {
                    let trimmed = self.input.trim();
                    participant.system_prompt = if trimmed.is_empty() {
                        None
                    } else {
                        Some(trimmed.to_string())
                    };
                }
            }
        }
    }

    /// Loads the current field's value into input.
    fn load_current_field(&mut self) {
        if self.editing_participant < self.participants.len() {
            let participant = &self.participants[self.editing_participant];
            self.input = match self.editing_field {
                ParticipantField::DisplayName => participant.display_name.clone(),
                ParticipantField::SystemPrompt => {
                    participant.system_prompt.clone().unwrap_or_default()
                }
            };
        }
    }

    /// Returns the participant currently being configured.
    pub fn current_editing_participant(&self) -> Option<&ParticipantDraft> {
        if self.step == DialogueBuilderStep::ConfigureParticipant {
            self.participants.get(self.editing_participant)
        } else {
            None
        }
    }

    /// Sets the error message.
    pub fn set_error(&mut self, error: impl Into<String>) {
        self.error = Some(error.into());
    }

    /// Validates the current step.
    pub fn validate(&self) -> Result<(), String> {
        match self.step {
            DialogueBuilderStep::Participants => {
                if self.participants.len() < 2 {
                    Err("At least 2 participants required".to_string())
                } else {
                    Ok(())
                }
            }
            _ => Ok(()),
        }
    }

    /// Returns whether the builder is complete and ready to start.
    pub fn is_complete(&self) -> bool {
        self.step == DialogueBuilderStep::Review && self.participants.len() >= 2
    }
}

/// Result of completing the dialogue builder.
#[derive(Debug, Clone)]
pub struct DialogueBuilderResult {
    /// Participants to create.
    pub participants: Vec<ParticipantDraft>,
    /// Selected dialogue mode.
    pub mode: DialogueMode,
    /// Initial prompt.
    pub initial_prompt: String,
}

impl From<&DialogueBuilderState> for DialogueBuilderResult {
    fn from(state: &DialogueBuilderState) -> Self {
        Self {
            participants: state.participants.clone(),
            mode: state.mode,
            initial_prompt: state.initial_prompt.clone(),
        }
    }
}
