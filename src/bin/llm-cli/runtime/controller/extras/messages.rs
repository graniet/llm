use crate::conversation::{ConversationMessage, MessageKind, MessageRole};
use crate::diff::parse_unified_diff;
use crate::runtime::AppStatus;
use crate::runtime::{DiffViewerState, InputMode, OverlayState, PagerState};

use super::super::AppController;

const PAGER_MIN_LINES: usize = 12;

impl AppController {
    pub fn edit_last_user(&mut self) -> bool {
        let content = if let Some(conv) = self.state.active_conversation_mut() {
            let idx = match conv
                .messages
                .iter()
                .rposition(|m| m.role == MessageRole::User)
            {
                Some(idx) => idx,
                None => return false,
            };
            let msg = conv.messages.remove(idx);
            self.record_snapshot();
            if let MessageKind::Text(text) = msg.kind {
                Some(text)
            } else {
                None
            }
        } else {
            None
        };
        if let Some(text) = content {
            self.state.input.set_text(text);
            self.state.input_mode = InputMode::Insert;
            return true;
        }
        false
    }

    pub fn copy_selected(&mut self) -> bool {
        let message = self.find_selected_message();
        if let Some(text) = message {
            if let Ok(mut clipboard) = arboard::Clipboard::new() {
                let _ = clipboard.set_text(text);
                return true;
            }
        }
        false
    }

    pub fn delete_selected(&mut self) -> bool {
        let id = match self.state.selected_message {
            Some(id) => id,
            None => return false,
        };
        if let Some(conv) = self.state.active_conversation_mut() {
            if let Some(idx) = conv.messages.iter().position(|m| m.id == id) {
                conv.messages.remove(idx);
                self.record_snapshot();
                return true;
            }
        }
        false
    }

    pub fn open_pager_for_selected(&mut self) -> bool {
        let text = match self.find_selected_message() {
            Some(text) => text,
            None => return false,
        };
        if text.lines().count() < PAGER_MIN_LINES {
            return false;
        }
        let title = "Message";
        self.state.overlay = OverlayState::Pager(PagerState::new(title, &text));
        true
    }

    pub fn open_diff_for_selected(&mut self) -> bool {
        let text = match self.find_selected_message() {
            Some(text) => text,
            None => return false,
        };
        match parse_unified_diff(&text) {
            Ok(diff) => {
                self.state.overlay = OverlayState::DiffViewer(DiffViewerState::new(diff));
                true
            }
            Err(err) => {
                self.set_status(AppStatus::Error(format!("diff parse: {err}")));
                false
            }
        }
    }

    pub fn select_prev_message(&mut self) -> bool {
        self.select_message_by_offset(-1)
    }

    pub fn select_next_message(&mut self) -> bool {
        self.select_message_by_offset(1)
    }

    pub fn select_first_message(&mut self) -> bool {
        self.select_message_at(0)
    }

    pub fn select_last_message(&mut self) -> bool {
        let len = match self.state.active_conversation() {
            Some(conv) => conv.messages.len(),
            None => return false,
        };
        if len == 0 {
            return false;
        }
        self.select_message_at(len.saturating_sub(1))
    }

    fn find_selected_message(&self) -> Option<String> {
        let conv = self.state.active_conversation()?;
        let target = self.state.selected_message?;
        let msg = conv.messages.iter().find(|m| m.id == target)?;
        Some(message_text(msg))
    }

    fn select_message_by_offset(&mut self, delta: i32) -> bool {
        let conv = match self.state.active_conversation() {
            Some(conv) => conv,
            None => return false,
        };
        let ids: Vec<_> = conv.messages.iter().map(|m| m.id).collect();
        if ids.is_empty() {
            return false;
        }
        let current = self
            .state
            .selected_message
            .and_then(|id| ids.iter().position(|m| *m == id));
        let start = match (current, delta.signum()) {
            (Some(idx), _) => idx,
            (None, -1) => ids.len().saturating_sub(1),
            (None, _) => 0,
        };
        let next = (start as i32 + delta).clamp(0, ids.len().saturating_sub(1) as i32) as usize;
        self.state.selected_message = Some(ids[next]);
        true
    }

    fn select_message_at(&mut self, index: usize) -> bool {
        let conv = match self.state.active_conversation() {
            Some(conv) => conv,
            None => return false,
        };
        let id = match conv.messages.get(index) {
            Some(msg) => msg.id,
            None => return false,
        };
        self.state.selected_message = Some(id);
        true
    }
}

pub(super) fn message_text(message: &ConversationMessage) -> String {
    match &message.kind {
        MessageKind::Text(text) => text.clone(),
        MessageKind::ToolCall(invocation) => invocation.arguments.clone(),
        MessageKind::ToolResult(result) => result.output.clone(),
        MessageKind::Error(text) => text.clone(),
    }
}
