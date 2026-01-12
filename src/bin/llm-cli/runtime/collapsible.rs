use std::collections::HashSet;

use crate::conversation::MessageId;

pub const TOOL_COLLAPSE_LINES: usize = 10;

#[derive(Debug, Default, Clone)]
pub struct CollapsibleState {
    expanded: HashSet<MessageId>,
}

impl CollapsibleState {
    pub fn is_expanded(&self, id: MessageId) -> bool {
        self.expanded.contains(&id)
    }

    pub fn toggle(&mut self, id: MessageId) -> bool {
        if self.expanded.remove(&id) {
            return false;
        }
        self.expanded.insert(id);
        true
    }

    pub fn set_all(&mut self, expanded: bool, ids: impl Iterator<Item = MessageId>) {
        self.expanded.clear();
        if expanded {
            self.expanded.extend(ids);
        }
    }
}
