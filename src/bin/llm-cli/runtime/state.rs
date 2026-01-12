use std::collections::HashMap;

use crate::config::AppConfig;
use crate::conversation::{Conversation, ConversationId, ConversationManager, MessageId};
use crate::history::BacktrackState;
use crate::input::{InputBuffer, InputHistory};
use crate::persistence::JsonConversationStore;
use crate::provider::{ProviderHandle, ProviderOverrides, ProviderRegistry};
use crate::skills::SkillCatalog;
use crate::terminal::TerminalCapabilities;

use super::overlay::OverlayState;
use super::scroll::ScrollState;
use super::status::AppStatus;
use super::PasteDetector;
use super::{AnimationState, CollapsibleState, StatusMetrics};

pub struct AppState {
    pub config: AppConfig,
    pub registry: ProviderRegistry,
    pub store: JsonConversationStore,
    pub conversations: ConversationManager,
    pub skills: SkillCatalog,
    pub input: InputBuffer,
    pub history: InputHistory,
    pub status: AppStatus,
    pub status_metrics: StatusMetrics,
    pub collapsible: CollapsibleState,
    pub backtrack: BacktrackState,
    pub overlay: OverlayState,
    pub scroll: ScrollState,
    pub provider_cache: HashMap<ConversationId, ProviderHandle>,
    pub selected_message: Option<MessageId>,
    pub input_mode: InputMode,
    pub focus: Focus,
    pub terminal_size: (u16, u16),
    pub terminal_caps: TerminalCapabilities,
    pub animation: AnimationState,
    pub paste_detector: PasteDetector,
    pub should_quit: bool,
    pub session_overrides: ProviderOverrides,
}

impl AppState {
    pub fn new(
        config: AppConfig,
        registry: ProviderRegistry,
        store: JsonConversationStore,
        terminal_caps: TerminalCapabilities,
    ) -> Self {
        Self {
            config,
            registry,
            store,
            conversations: ConversationManager::new(),
            skills: SkillCatalog::default(),
            input: InputBuffer::default(),
            history: InputHistory::default(),
            status: AppStatus::Idle,
            status_metrics: StatusMetrics::default(),
            collapsible: CollapsibleState::default(),
            backtrack: BacktrackState::default(),
            overlay: OverlayState::None,
            scroll: ScrollState::default(),
            provider_cache: HashMap::new(),
            selected_message: None,
            input_mode: InputMode::Insert,
            focus: Focus::Input,
            terminal_size: (0, 0),
            terminal_caps,
            animation: AnimationState::default(),
            paste_detector: PasteDetector::default(),
            should_quit: false,
            session_overrides: ProviderOverrides::default(),
        }
    }

    pub fn active_conversation(&self) -> Option<&Conversation> {
        self.conversations.active()
    }

    pub fn active_conversation_mut(&mut self) -> Option<&mut Conversation> {
        self.conversations.active_mut()
    }

    pub fn active_conversation_id(&self) -> Option<ConversationId> {
        self.conversations.active_id()
    }

    pub fn mark_dirty(&mut self) {
        if let Some(conv) = self.active_conversation_mut() {
            conv.dirty = true;
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum InputMode {
    Insert,
    Normal,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Focus {
    Input,
    Messages,
}
