use std::collections::VecDeque;

use chrono::Utc;

use crate::provider::ProviderId;

use super::id::ConversationId;
use super::state::Conversation;

#[derive(Debug)]
pub struct ConversationManager {
    conversations: VecDeque<Conversation>,
    active_id: Option<ConversationId>,
}

impl ConversationManager {
    pub fn new() -> Self {
        Self {
            conversations: VecDeque::new(),
            active_id: None,
        }
    }

    pub fn add(&mut self, conversation: Conversation) {
        let id = conversation.id;
        self.conversations.push_front(conversation);
        self.active_id = Some(id);
    }

    pub fn list(&self) -> impl Iterator<Item = &Conversation> {
        self.conversations.iter()
    }

    pub fn active(&self) -> Option<&Conversation> {
        let id = self.active_id?;
        self.conversations.iter().find(|c| c.id == id)
    }

    pub fn active_mut(&mut self) -> Option<&mut Conversation> {
        let id = self.active_id?;
        self.conversations.iter_mut().find(|c| c.id == id)
    }

    pub fn set_active(&mut self, id: ConversationId) -> bool {
        if self.conversations.iter().any(|c| c.id == id) {
            self.active_id = Some(id);
            true
        } else {
            false
        }
    }

    pub fn new_conversation(
        &mut self,
        provider_id: ProviderId,
        model: Option<String>,
        system_prompt: Option<String>,
    ) -> ConversationId {
        let conversation = Conversation::new(provider_id, model, system_prompt);
        let id = conversation.id;
        self.add(conversation);
        id
    }

    pub fn fork_conversation(&mut self, id: ConversationId) -> Option<ConversationId> {
        let mut cloned = self.conversations.iter().find(|c| c.id == id)?.clone();
        let now = Utc::now();
        cloned.id = ConversationId::new();
        cloned.parent_id = Some(id);
        cloned.created_at = now;
        cloned.updated_at = now;
        cloned.title = format!("Copy of {}", cloned.title);
        cloned.dirty = true;
        self.add(cloned);
        self.active_id
    }

    pub fn active_id(&self) -> Option<ConversationId> {
        self.active_id
    }
}
