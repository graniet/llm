#[derive(Debug, Default)]
pub struct InputHistory {
    entries: Vec<String>,
    cursor: Option<usize>,
    draft: Option<String>,
}

impl InputHistory {
    pub fn record(&mut self, entry: String) {
        if entry.trim().is_empty() {
            return;
        }
        if self
            .entries
            .last()
            .map(|last| last == &entry)
            .unwrap_or(false)
        {
            return;
        }
        self.entries.push(entry);
        self.cursor = None;
        self.draft = None;
    }

    pub fn previous(&mut self, current: &str) -> Option<String> {
        if self.entries.is_empty() {
            return None;
        }
        let next_cursor = match self.cursor {
            None => {
                self.draft = Some(current.to_string());
                self.entries.len().saturating_sub(1)
            }
            Some(idx) => idx.saturating_sub(1),
        };
        self.cursor = Some(next_cursor);
        self.entries.get(next_cursor).cloned()
    }

    pub fn next(&mut self) -> Option<String> {
        let idx = self.cursor?;
        if idx + 1 < self.entries.len() {
            let next_idx = idx + 1;
            self.cursor = Some(next_idx);
            return self.entries.get(next_idx).cloned();
        }
        self.cursor = None;
        self.draft.take()
    }
}
