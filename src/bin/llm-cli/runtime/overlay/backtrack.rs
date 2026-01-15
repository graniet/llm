use crate::history::SnapshotSummary;

#[derive(Debug, Clone)]
pub struct BacktrackOverlayState {
    pub entries: Vec<SnapshotSummary>,
    pub selected: usize,
}

impl BacktrackOverlayState {
    pub fn new(entries: Vec<SnapshotSummary>) -> Self {
        let selected = entries.len().saturating_sub(1);
        Self { entries, selected }
    }

    pub fn selected_entry(&self) -> Option<&SnapshotSummary> {
        self.entries.get(self.selected)
    }

    pub fn next(&mut self) {
        if !self.entries.is_empty() {
            self.selected = (self.selected + 1).min(self.entries.len().saturating_sub(1));
        }
    }

    pub fn prev(&mut self) {
        if self.selected > 0 {
            self.selected = self.selected.saturating_sub(1);
        }
    }
}
