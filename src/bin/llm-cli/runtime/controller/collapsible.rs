use crate::conversation::MessageKind;
use crate::runtime::TOOL_COLLAPSE_LINES;

use super::AppController;

impl AppController {
    pub fn toggle_selected_tool_output(&mut self) -> bool {
        let id = match self.state.selected_message {
            Some(id) => id,
            None => return false,
        };
        let should_toggle = {
            let conv = match self.state.active_conversation() {
                Some(conv) => conv,
                None => return false,
            };
            let msg = match conv.messages.iter().find(|m| m.id == id) {
                Some(msg) => msg,
                None => return false,
            };
            match &msg.kind {
                MessageKind::ToolResult(result) => {
                    result.output.lines().count() > TOOL_COLLAPSE_LINES
                }
                _ => false,
            }
        };
        if !should_toggle {
            return false;
        }
        self.state.collapsible.toggle(id);
        true
    }

    pub fn toggle_all_tool_outputs(&mut self) -> bool {
        let conv = match self.state.active_conversation() {
            Some(conv) => conv,
            None => return false,
        };
        let ids: Vec<_> = conv
            .messages
            .iter()
            .filter_map(|msg| {
                let MessageKind::ToolResult(result) = &msg.kind else {
                    return None;
                };
                if result.output.lines().count() <= TOOL_COLLAPSE_LINES {
                    return None;
                }
                Some(msg.id)
            })
            .collect();
        if ids.is_empty() {
            return false;
        }
        let all_expanded = ids.iter().all(|id| self.state.collapsible.is_expanded(*id));
        self.state
            .collapsible
            .set_all(!all_expanded, ids.into_iter());
        true
    }
}
