use std::fs;
use std::path::{Component, Path, PathBuf};

use thiserror::Error;

use super::{DiffFile, DiffHunk, DiffLine, DiffView, HunkDecision, LineKind};

#[derive(Debug, Error)]
pub enum DiffApplyError {
    #[error("no accepted hunks to apply")]
    NothingToApply,
    #[error("unsupported diff path: {0}")]
    InvalidPath(String),
    #[error("context mismatch while applying diff")]
    ContextMismatch,
    #[error("missing file header for diff")]
    MissingFile,
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub fn apply_diff(diff: &DiffView, base_dir: &Path) -> Result<Vec<PathBuf>, DiffApplyError> {
    let mut applied = Vec::new();
    let mut any = false;
    for file in &diff.files {
        if file
            .hunks
            .iter()
            .any(|h| h.decision == HunkDecision::Accepted)
        {
            any = true;
            if let Some(path) = apply_file(file, base_dir)? {
                applied.push(path);
            }
        }
    }
    if !any {
        return Err(DiffApplyError::NothingToApply);
    }
    Ok(applied)
}

fn apply_file(file: &DiffFile, base_dir: &Path) -> Result<Option<PathBuf>, DiffApplyError> {
    let target_path = resolve_target_path(file)?;
    let full_path = base_dir.join(&target_path);
    let (original_lines, had_newline) = read_lines(&full_path)?;

    let mut output = Vec::new();
    let mut index = 0usize;
    for hunk in file
        .hunks
        .iter()
        .filter(|h| h.decision == HunkDecision::Accepted)
    {
        let start = hunk.old_start.saturating_sub(1);
        if start > original_lines.len() {
            return Err(DiffApplyError::ContextMismatch);
        }
        output.extend_from_slice(&original_lines[index..start]);
        index = start;
        apply_hunk_lines(hunk, &original_lines, &mut index, &mut output)?;
    }
    output.extend_from_slice(&original_lines[index..]);

    write_with_backup(&full_path, &output, had_newline)?;
    Ok(Some(full_path))
}

fn resolve_target_path(file: &DiffFile) -> Result<PathBuf, DiffApplyError> {
    let path = if file.new_path != "/dev/null" {
        &file.new_path
    } else if file.old_path != "/dev/null" {
        &file.old_path
    } else {
        return Err(DiffApplyError::MissingFile);
    };
    sanitize_path(path)
}

fn sanitize_path(raw: &str) -> Result<PathBuf, DiffApplyError> {
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        return Err(DiffApplyError::InvalidPath(raw.to_string()));
    }
    if path.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(DiffApplyError::InvalidPath(raw.to_string()));
    }
    Ok(path)
}

fn read_lines(path: &Path) -> Result<(Vec<String>, bool), DiffApplyError> {
    match fs::read_to_string(path) {
        Ok(contents) => {
            let had_newline = contents.ends_with('\n');
            Ok((
                contents.lines().map(|line| line.to_string()).collect(),
                had_newline,
            ))
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok((Vec::new(), false)),
        Err(err) => Err(err.into()),
    }
}

fn apply_hunk_lines(
    hunk: &DiffHunk,
    original: &[String],
    index: &mut usize,
    output: &mut Vec<String>,
) -> Result<(), DiffApplyError> {
    for line in &hunk.lines {
        match line.kind {
            LineKind::Context => {
                ensure_line(original, *index, line)?;
                output.push(original[*index].clone());
                *index += 1;
            }
            LineKind::Remove => {
                ensure_line(original, *index, line)?;
                *index += 1;
            }
            LineKind::Add => output.push(line.content.clone()),
        }
    }
    Ok(())
}

fn ensure_line(original: &[String], index: usize, line: &DiffLine) -> Result<(), DiffApplyError> {
    if original.get(index).map(|v| v.as_str()) == Some(line.content.as_str()) {
        Ok(())
    } else {
        Err(DiffApplyError::ContextMismatch)
    }
}

fn write_with_backup(
    path: &Path,
    lines: &[String],
    had_newline: bool,
) -> Result<(), DiffApplyError> {
    if path.exists() {
        let backup = PathBuf::from(format!("{}.bak", path.display()));
        fs::copy(path, backup)?;
    }
    let mut contents = lines.join("\n");
    if had_newline {
        contents.push('\n');
    }
    fs::write(path, contents)?;
    Ok(())
}
