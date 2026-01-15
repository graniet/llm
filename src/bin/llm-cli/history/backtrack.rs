use std::collections::HashMap;

use chrono::{DateTime, Utc};

use crate::conversation::{Conversation, ConversationId};

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct SnapshotId(u64);

impl SnapshotId {
    pub const fn new(value: u64) -> Self {
        Self(value)
    }
}

#[derive(Debug, Clone)]
pub struct SnapshotSummary {
    pub id: SnapshotId,
    pub title: String,
    pub created_at: DateTime<Utc>,
    pub message_count: usize,
}

#[derive(Debug, Clone)]
pub struct Snapshot {
    pub id: SnapshotId,
    pub conversation: Conversation,
    pub created_at: DateTime<Utc>,
    pub message_count: usize,
}

impl Snapshot {
    fn new(id: SnapshotId, conversation: Conversation) -> Self {
        let message_count = conversation.messages.len();
        Self {
            id,
            created_at: Utc::now(),
            message_count,
            conversation,
        }
    }

    fn summary(&self) -> SnapshotSummary {
        SnapshotSummary {
            id: self.id,
            title: self.conversation.title.clone(),
            created_at: self.created_at,
            message_count: self.message_count,
        }
    }

    fn matches(&self, conversation: &Conversation) -> bool {
        self.message_count == conversation.messages.len()
            && self.conversation.updated_at == conversation.updated_at
    }
}

#[derive(Debug, Default)]
pub struct BacktrackState {
    snapshots: HashMap<ConversationId, Vec<Snapshot>>,
    next_id: u64,
}

impl BacktrackState {
    pub fn record(&mut self, conversation: &Conversation) -> Option<SnapshotSummary> {
        let snapshots = self.snapshots.entry(conversation.id).or_default();
        if snapshots
            .last()
            .map(|snap| snap.matches(conversation))
            .unwrap_or(false)
        {
            return None;
        }
        let id = SnapshotId::new(self.next_id);
        self.next_id = self.next_id.saturating_add(1);
        let snapshot = Snapshot::new(id, conversation.clone());
        snapshots.push(snapshot);
        snapshots.last().map(Snapshot::summary)
    }

    pub fn list(&self, conversation_id: ConversationId) -> Vec<SnapshotSummary> {
        self.snapshots
            .get(&conversation_id)
            .map(|items| items.iter().map(Snapshot::summary).collect())
            .unwrap_or_default()
    }

    pub fn get(
        &self,
        conversation_id: ConversationId,
        snapshot_id: SnapshotId,
    ) -> Option<&Snapshot> {
        self.snapshots
            .get(&conversation_id)
            .and_then(|items| items.iter().find(|snap| snap.id == snapshot_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::{Conversation, MessageKind, MessageRole};
    use crate::provider::ProviderId;

    fn sample_conversation() -> Conversation {
        let provider = ProviderId::new("openai");
        let mut conv = Conversation::new(provider, None, None);
        conv.push_message(crate::conversation::ConversationMessage::new(
            MessageRole::User,
            MessageKind::Text("hello".to_string()),
        ));
        conv
    }

    #[test]
    fn records_unique_snapshots() {
        let mut backtrack = BacktrackState::default();
        let mut conv = sample_conversation();
        let first = backtrack.record(&conv);
        assert!(first.is_some());
        let second = backtrack.record(&conv);
        assert!(second.is_none());
        conv.updated_at = Utc::now();
        let third = backtrack.record(&conv);
        assert!(third.is_some());
    }
}
