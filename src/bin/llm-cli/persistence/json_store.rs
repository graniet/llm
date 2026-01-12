use std::fs;
use std::path::{Path, PathBuf};

use crate::conversation::{Conversation, ConversationId};

use super::error::PersistenceError;

#[derive(Debug, Clone)]
pub struct JsonConversationStore {
    dir: PathBuf,
}

impl JsonConversationStore {
    pub fn new(dir: PathBuf) -> Self {
        Self { dir }
    }

    pub fn load_all(&self) -> Result<Vec<Conversation>, PersistenceError> {
        let mut items = Vec::new();
        if !self.dir.exists() {
            return Ok(items);
        }
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("json") {
                continue;
            }
            if let Ok(conv) = self.load_path(&path) {
                items.push(conv);
            }
        }
        Ok(items)
    }

    pub fn save(&self, conversation: &Conversation) -> Result<(), PersistenceError> {
        fs::create_dir_all(&self.dir)?;
        let payload = serde_json::to_vec_pretty(conversation)?;
        let path = self.path_for(conversation.id);
        fs::write(path, payload)?;
        Ok(())
    }

    fn load_path(&self, path: &Path) -> Result<Conversation, PersistenceError> {
        let data = fs::read(path)?;
        Ok(serde_json::from_slice(&data)?)
    }

    fn path_for(&self, id: ConversationId) -> PathBuf {
        self.dir.join(format!("{id}.json"))
    }
}

#[cfg(test)]
mod tests {
    use std::time::{SystemTime, UNIX_EPOCH};

    use super::*;
    use crate::conversation::{ConversationMessage, MessageKind, MessageRole};
    use crate::provider::ProviderId;

    fn temp_store_dir() -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let mut dir = std::env::temp_dir();
        dir.push(format!("llm-cli-test-{nanos}"));
        dir
    }

    #[test]
    fn save_and_load_round_trip() {
        let dir = temp_store_dir();
        let store = JsonConversationStore::new(dir.clone());
        let mut conversation = Conversation::new(
            ProviderId::new("openai"),
            Some("gpt-4o-mini".to_string()),
            Some("system".to_string()),
        );
        conversation.push_message(ConversationMessage::new(
            MessageRole::User,
            MessageKind::Text("hello".to_string()),
        ));
        store.save(&conversation).expect("save conversation");

        let loaded = store
            .load_all()
            .expect("load conversations")
            .into_iter()
            .find(|conv| conv.id == conversation.id)
            .expect("missing conversation");
        assert_eq!(loaded.messages.len(), conversation.messages.len());

        let _ = fs::remove_dir_all(dir);
    }
}
