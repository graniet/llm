use chrono::Utc;

use crate::conversation::ConversationId;
use crate::history::SnapshotId;
use crate::runtime::{BacktrackOverlayState, OverlayState, PickerItem, PickerState};

use super::AppController;

impl AppController {
    pub fn record_snapshot(&mut self) -> bool {
        let Some(conv) = self.state.active_conversation().cloned() else {
            return false;
        };
        self.state.backtrack.record(&conv).is_some()
    }

    pub fn open_backtrack(&mut self) -> bool {
        let conv_id = match self.state.active_conversation_id() {
            Some(id) => id,
            None => return false,
        };
        if let Some(conv) = self.state.active_conversation().cloned() {
            self.state.backtrack.record(&conv);
        }
        let entries = self.state.backtrack.list(conv_id);
        if entries.is_empty() {
            self.set_status(crate::runtime::AppStatus::Error(
                "no snapshots available".to_string(),
            ));
            return false;
        }
        self.state.overlay = OverlayState::Backtrack(BacktrackOverlayState::new(entries));
        true
    }

    pub fn restore_snapshot(&mut self, snapshot_id: SnapshotId) -> bool {
        let conv_id = match self.state.active_conversation_id() {
            Some(id) => id,
            None => return false,
        };
        let snapshot = match self.state.backtrack.get(conv_id, snapshot_id) {
            Some(snapshot) => snapshot.clone(),
            None => return false,
        };
        let mut conv = snapshot.conversation.clone();
        let parent = conv.id;
        let now = Utc::now();
        conv.id = ConversationId::new();
        conv.parent_id = Some(parent);
        conv.created_at = now;
        conv.updated_at = now;
        conv.title = format!("Branch of {}", conv.title);
        conv.dirty = true;
        self.state.conversations.add(conv);
        self.state.scroll.reset();
        self.record_snapshot();
        true
    }

    pub fn open_branches(&mut self) -> bool {
        let items = self
            .state
            .conversations
            .list()
            .filter(|conv| conv.parent_id.is_some())
            .map(|conv| PickerItem {
                id: conv.id.to_string(),
                label: conv.title.clone(),
                meta: conv.parent_id.map(|id| id.to_string()),
                badges: vec!["branch".to_string()],
            })
            .collect();
        self.state.overlay = OverlayState::ConversationPicker(PickerState::new("Branches", items));
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AppConfig;
    use crate::conversation::{ConversationMessage, MessageKind, MessageRole};
    use crate::persistence::JsonConversationStore;
    use crate::runtime::{AppState, StreamManager};
    use crate::terminal::TerminalCapabilities;
    use crate::tools::{ToolContext, ToolRegistry};
    use crate::{config::ConfigPaths, provider::ProviderRegistry};

    fn build_controller() -> AppController {
        let config = AppConfig::default();
        let registry = ProviderRegistry::from_config(&config.providers);
        let store = JsonConversationStore::new(std::env::temp_dir().join("llm-test-conv"));
        let terminal_caps = TerminalCapabilities::detect();
        let state = AppState::new(config.clone(), registry, store, terminal_caps);
        let (tx, _rx) = tokio::sync::mpsc::channel(1);
        let stream_manager = StreamManager::new(tx.clone());
        let tool_registry = ToolRegistry::from_config(&config.tools);
        let tool_context = ToolContext {
            allowed_paths: config.tools.allowed_paths.clone(),
            timeout_ms: config.tools.timeout_ms,
            working_dir: ".".to_string(),
        };
        let config_paths =
            ConfigPaths::resolve(Some(std::env::temp_dir().join("llm-test-config.toml"))).unwrap();
        let params = crate::runtime::controller::AppControllerParams {
            state,
            stream_manager,
            event_sender: tx,
            tool_registry,
            tool_context,
            config_paths,
        };
        AppController::new(params)
    }

    #[test]
    fn restores_snapshot_as_branch() {
        let mut controller = build_controller();
        controller
            .state
            .conversations
            .new_conversation("openai".into(), None, None);
        let conv_id = controller.state.active_conversation_id().unwrap();
        if let Some(conv) = controller.state.active_conversation_mut() {
            conv.push_message(ConversationMessage::new(
                MessageRole::User,
                MessageKind::Text("hello".to_string()),
            ));
        }
        let conv = controller.state.active_conversation().cloned().unwrap();
        let snapshot = controller.state.backtrack.record(&conv).unwrap();
        assert!(controller.restore_snapshot(snapshot.id));
        let new_id = controller.state.active_conversation_id().unwrap();
        assert_ne!(new_id, conv_id);
        let new_conv = controller.state.active_conversation().unwrap();
        assert_eq!(new_conv.parent_id, Some(conv_id));
    }
}
