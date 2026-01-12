use thiserror::Error;

use super::{DiffFile, DiffHunk, DiffLine, DiffView, HunkDecision, LineKind};

#[derive(Debug, Error)]
pub enum DiffParseError {
    #[error("diff is empty")]
    EmptyDiff,
    #[error("missing file header before hunk")]
    MissingFileHeader,
    #[error("invalid hunk header: {0}")]
    InvalidHunkHeader(String),
}

pub fn parse_unified_diff(input: &str) -> Result<DiffView, DiffParseError> {
    let mut view = DiffView::default();
    let mut current_file: Option<usize> = None;
    let mut pending_old: Option<String> = None;

    for line in input.lines() {
        if let Some(old) = line.strip_prefix("--- ") {
            pending_old = Some(clean_path(old.trim()));
            continue;
        }
        if let Some(new) = line.strip_prefix("+++ ") {
            let old = pending_old
                .take()
                .ok_or(DiffParseError::MissingFileHeader)?;
            let new = clean_path(new.trim());
            view.files.push(DiffFile {
                old_path: old,
                new_path: new,
                hunks: Vec::new(),
            });
            current_file = Some(view.files.len().saturating_sub(1));
            continue;
        }
        if line.starts_with("@@") {
            let (old_start, new_start, header) = parse_hunk_header(line)?;
            let idx = current_file.ok_or(DiffParseError::MissingFileHeader)?;
            view.files[idx].hunks.push(DiffHunk {
                header,
                old_start,
                new_start,
                lines: Vec::new(),
                decision: HunkDecision::Pending,
            });
            continue;
        }
        let Some(idx) = current_file else {
            continue;
        };
        if view.files[idx].hunks.is_empty() {
            continue;
        }
        if let Some((kind, content)) = parse_diff_line(line) {
            let hunk_idx = view.files[idx].hunks.len().saturating_sub(1);
            view.files[idx].hunks[hunk_idx]
                .lines
                .push(DiffLine { kind, content });
        }
    }

    if view.files.is_empty() {
        return Err(DiffParseError::EmptyDiff);
    }
    Ok(view)
}

fn parse_hunk_header(line: &str) -> Result<(usize, usize, String), DiffParseError> {
    let trimmed = line.trim();
    let start = trimmed
        .find("@@")
        .ok_or_else(|| DiffParseError::InvalidHunkHeader(line.to_string()))?;
    let rest = &trimmed[start + 2..];
    let end = rest
        .find("@@")
        .ok_or_else(|| DiffParseError::InvalidHunkHeader(line.to_string()))?;
    let body = rest[..end].trim();
    let parts: Vec<&str> = body.split_whitespace().collect();
    if parts.len() < 2 {
        return Err(DiffParseError::InvalidHunkHeader(line.to_string()));
    }
    let old_start = parse_range(parts[0], '-')?;
    let new_start = parse_range(parts[1], '+')?;
    Ok((old_start, new_start, line.to_string()))
}

fn parse_range(token: &str, prefix: char) -> Result<usize, DiffParseError> {
    let value = token
        .strip_prefix(prefix)
        .ok_or_else(|| DiffParseError::InvalidHunkHeader(token.to_string()))?;
    let number = value.split(',').next().unwrap_or(value);
    number
        .parse::<usize>()
        .map_err(|_| DiffParseError::InvalidHunkHeader(token.to_string()))
}

fn parse_diff_line(line: &str) -> Option<(LineKind, String)> {
    let mut chars = line.chars();
    let prefix = chars.next()?;
    let kind = match prefix {
        '+' => LineKind::Add,
        '-' => LineKind::Remove,
        ' ' => LineKind::Context,
        _ => return None,
    };
    Some((kind, chars.collect()))
}

fn clean_path(raw: &str) -> String {
    raw.trim_start_matches("a/")
        .trim_start_matches("b/")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_diff() {
        let raw = "--- a/foo.txt\n+++ b/foo.txt\n@@ -1,2 +1,2 @@\n-old\n+new\n";
        let diff = parse_unified_diff(raw).unwrap();
        assert_eq!(diff.files.len(), 1);
        assert_eq!(diff.files[0].hunks.len(), 1);
    }
}
