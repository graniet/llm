use crate::diff::{DiffHunk, DiffView, HunkDecision};

#[derive(Debug, Clone)]
pub struct DiffViewerState {
    pub diff: DiffView,
    pub file_index: usize,
    pub hunk_index: usize,
    pub scroll: u16,
}

impl DiffViewerState {
    pub fn new(diff: DiffView) -> Self {
        Self {
            diff,
            file_index: 0,
            hunk_index: 0,
            scroll: 0,
        }
    }

    pub fn current_hunk(&self) -> Option<&DiffHunk> {
        self.diff
            .files
            .get(self.file_index)
            .and_then(|file| file.hunks.get(self.hunk_index))
    }

    pub fn current_hunk_mut(&mut self) -> Option<&mut DiffHunk> {
        self.diff
            .files
            .get_mut(self.file_index)
            .and_then(|file| file.hunks.get_mut(self.hunk_index))
    }

    pub fn next_hunk(&mut self) {
        if let Some((file_idx, hunk_idx)) =
            next_hunk_index(&self.diff, self.file_index, self.hunk_index)
        {
            self.file_index = file_idx;
            self.hunk_index = hunk_idx;
            self.scroll = 0;
        }
    }

    pub fn prev_hunk(&mut self) {
        if let Some((file_idx, hunk_idx)) =
            prev_hunk_index(&self.diff, self.file_index, self.hunk_index)
        {
            self.file_index = file_idx;
            self.hunk_index = hunk_idx;
            self.scroll = 0;
        }
    }

    pub fn accept_current(&mut self) {
        if let Some(hunk) = self.current_hunk_mut() {
            hunk.set_decision(HunkDecision::Accepted);
        }
    }

    pub fn reject_current(&mut self) {
        if let Some(hunk) = self.current_hunk_mut() {
            hunk.set_decision(HunkDecision::Rejected);
        }
    }

    pub fn skip_current(&mut self) {
        if let Some(hunk) = self.current_hunk_mut() {
            hunk.set_decision(HunkDecision::Skipped);
        }
    }

    pub fn accept_all(&mut self) {
        for file in &mut self.diff.files {
            for hunk in &mut file.hunks {
                if hunk.decision == HunkDecision::Pending {
                    hunk.set_decision(HunkDecision::Accepted);
                }
            }
        }
    }

    pub fn scroll_up(&mut self, lines: u16) {
        self.scroll = self.scroll.saturating_sub(lines);
    }

    pub fn scroll_down(&mut self, lines: u16, height: u16) {
        let max = self.max_scroll(height);
        self.scroll = (self.scroll + lines).min(max);
    }

    pub fn max_scroll(&self, height: u16) -> u16 {
        let total = self.current_hunk_line_count();
        total.saturating_sub(height)
    }

    fn current_hunk_line_count(&self) -> u16 {
        self.current_hunk()
            .map(|hunk| hunk.lines.len() as u16 + 1)
            .unwrap_or(0)
    }
}

fn next_hunk_index(
    diff: &DiffView,
    file_index: usize,
    hunk_index: usize,
) -> Option<(usize, usize)> {
    let file = diff.files.get(file_index)?;
    if hunk_index + 1 < file.hunks.len() {
        return Some((file_index, hunk_index + 1));
    }
    let mut next_file = file_index + 1;
    while let Some(file) = diff.files.get(next_file) {
        if !file.hunks.is_empty() {
            return Some((next_file, 0));
        }
        next_file += 1;
    }
    None
}

fn prev_hunk_index(
    diff: &DiffView,
    file_index: usize,
    hunk_index: usize,
) -> Option<(usize, usize)> {
    if hunk_index > 0 {
        return Some((file_index, hunk_index - 1));
    }
    let mut prev_file = file_index.checked_sub(1)?;
    loop {
        let file = diff.files.get(prev_file)?;
        if !file.hunks.is_empty() {
            let idx = file.hunks.len().saturating_sub(1);
            return Some((prev_file, idx));
        }
        prev_file = prev_file.checked_sub(1)?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::{DiffFile, DiffHunk, DiffLine, DiffView, HunkDecision, LineKind};

    fn sample_diff() -> DiffView {
        let hunk = DiffHunk {
            header: "@@ -1,1 +1,1 @@".to_string(),
            old_start: 1,
            new_start: 1,
            lines: vec![DiffLine {
                kind: LineKind::Add,
                content: "hello".to_string(),
            }],
            decision: HunkDecision::Pending,
        };
        DiffView {
            files: vec![DiffFile {
                old_path: "a.txt".to_string(),
                new_path: "a.txt".to_string(),
                hunks: vec![hunk],
            }],
        }
    }

    #[test]
    fn accepts_current_hunk() {
        let mut state = DiffViewerState::new(sample_diff());
        state.accept_current();
        let hunk = state.current_hunk().unwrap();
        assert_eq!(hunk.decision, HunkDecision::Accepted);
    }

    #[test]
    fn skips_empty_files_when_navigating() {
        let mut diff = sample_diff();
        diff.files.insert(
            0,
            DiffFile {
                old_path: "empty.txt".to_string(),
                new_path: "empty.txt".to_string(),
                hunks: Vec::new(),
            },
        );
        let mut state = DiffViewerState::new(diff);
        state.next_hunk();
        assert_eq!(state.file_index, 1);
    }
}
