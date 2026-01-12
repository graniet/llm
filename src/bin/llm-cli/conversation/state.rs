use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::provider::ProviderId;

use super::id::ConversationId;
use super::message::{ConversationMessage, MessageKind, MessageRole};

const TITLE_MAX_CHARS: usize = 48;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conversation {
    pub id: ConversationId,
    #[serde(default)]
    pub parent_id: Option<ConversationId>,
    pub title: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub provider_id: ProviderId,
    pub model: Option<String>,
    pub system_prompt: Option<String>,
    pub messages: Vec<ConversationMessage>,
    pub dirty: bool,
}

impl Conversation {
    pub fn new(
        provider_id: ProviderId,
        model: Option<String>,
        system_prompt: Option<String>,
    ) -> Self {
        let now = Utc::now();
        Self {
            id: ConversationId::new(),
            parent_id: None,
            title: "New conversation".to_string(),
            created_at: now,
            updated_at: now,
            provider_id,
            model,
            system_prompt,
            messages: Vec::new(),
            dirty: false,
        }
    }

    pub fn push_message(&mut self, message: ConversationMessage) {
        self.messages.push(message);
        self.updated_at = Utc::now();
        self.dirty = true;
    }

    pub fn title_from_first_user(&mut self) {
        let first_user = self
            .messages
            .iter()
            .find(|msg| msg.role == MessageRole::User);
        if let Some(msg) = first_user {
            if let MessageKind::Text(text) = &msg.kind {
                self.title = truncate_title(text);
            }
        }
    }
}

fn truncate_title(text: &str) -> String {
    let trimmed = text.trim();
    let mut chars = trimmed.chars();
    let mut title = chars.by_ref().take(TITLE_MAX_CHARS).collect::<String>();
    if chars.next().is_some() {
        title.push_str("...");
    }
    if title.is_empty() {
        "Conversation".to_string()
    } else {
        title
    }
}
