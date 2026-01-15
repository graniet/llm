mod apply;
mod parser;
mod types;

pub use apply::apply_diff;
pub use parser::parse_unified_diff;
pub use types::{DiffFile, DiffHunk, DiffLine, DiffView, HunkDecision, LineKind};
