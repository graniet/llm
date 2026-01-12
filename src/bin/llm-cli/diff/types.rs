#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum LineKind {
    Context,
    Add,
    Remove,
}

#[derive(Debug, Clone)]
pub struct DiffLine {
    pub kind: LineKind,
    pub content: String,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum HunkDecision {
    Pending,
    Accepted,
    Rejected,
    Skipped,
}

#[derive(Debug, Clone)]
pub struct DiffHunk {
    pub header: String,
    pub old_start: usize,
    pub new_start: usize,
    pub lines: Vec<DiffLine>,
    pub decision: HunkDecision,
}

#[derive(Debug, Clone)]
pub struct DiffFile {
    pub old_path: String,
    pub new_path: String,
    pub hunks: Vec<DiffHunk>,
}

#[derive(Debug, Clone, Default)]
pub struct DiffView {
    pub files: Vec<DiffFile>,
}

impl DiffHunk {
    pub fn set_decision(&mut self, decision: HunkDecision) {
        self.decision = decision;
    }
}
