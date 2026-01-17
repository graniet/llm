//! Directory traversal for the ls tool.

use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};

use crate::tools::error::ToolError;

use super::types::{format_entry, truncate_name, DirEntry, EntryKind};

/// Collect directory entries with breadth-first traversal.
pub fn collect_entries(root: &Path, max_depth: usize) -> Result<Vec<DirEntry>, ToolError> {
    let mut entries = Vec::new();
    let mut queue: VecDeque<(PathBuf, PathBuf, usize)> = VecDeque::new();
    queue.push_back((root.to_path_buf(), PathBuf::new(), max_depth));

    while let Some((current_dir, prefix, remaining_depth)) = queue.pop_front() {
        let mut dir_entries = read_dir_sorted(&current_dir)?;

        for (entry_path, file_name, kind) in dir_entries.drain(..) {
            let relative_path = if prefix.as_os_str().is_empty() {
                PathBuf::from(&file_name)
            } else {
                prefix.join(&file_name)
            };

            let display_depth = prefix.components().count();
            let sort_key = truncate_name(&relative_path.to_string_lossy());
            let display_name = truncate_name(&file_name);

            entries.push(DirEntry {
                name: sort_key,
                display_name,
                depth: display_depth,
                kind,
            });

            if kind == EntryKind::Directory && remaining_depth > 1 {
                queue.push_back((entry_path, relative_path, remaining_depth - 1));
            }
        }
    }

    Ok(entries)
}

/// Read directory entries sorted by name.
fn read_dir_sorted(dir: &Path) -> Result<Vec<(PathBuf, String, EntryKind)>, ToolError> {
    let read_dir =
        fs::read_dir(dir).map_err(|e| ToolError::Execution(format!("Failed to read dir: {e}")))?;

    let mut entries: Vec<(PathBuf, String, EntryKind)> = Vec::new();

    for entry in read_dir.flatten() {
        let path = entry.path();
        let file_name = entry.file_name().to_string_lossy().to_string();

        // Use symlink_metadata to detect symlinks
        let kind = match entry.metadata() {
            Ok(meta) => EntryKind::from(meta.file_type()),
            Err(_) => EntryKind::Other,
        };

        entries.push((path, file_name, kind));
    }

    entries.sort_by(|a, b| a.1.cmp(&b.1));
    Ok(entries)
}

/// Slice entries for pagination.
pub fn slice_entries(
    entries: Vec<DirEntry>,
    offset: usize,
    limit: usize,
) -> Result<Vec<String>, ToolError> {
    if entries.is_empty() {
        return Ok(Vec::new());
    }

    let start = offset - 1;
    if start >= entries.len() {
        return Err(ToolError::RespondToModel(
            "offset exceeds entry count".to_string(),
        ));
    }

    let end = (start + limit).min(entries.len());
    let mut formatted: Vec<String> = entries[start..end].iter().map(format_entry).collect();

    if end < entries.len() {
        formatted.push(format!("More than {} entries found", limit));
    }

    Ok(formatted)
}
