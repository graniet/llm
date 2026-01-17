//! Types for the ls tool.

use std::fs::FileType;

use serde::Deserialize;

/// Default offset (1-indexed).
pub const DEFAULT_OFFSET: usize = 1;

/// Default limit.
pub const DEFAULT_LIMIT: usize = 25;

/// Default depth.
pub const DEFAULT_DEPTH: usize = 2;

/// Maximum entry name length.
pub const MAX_ENTRY_LENGTH: usize = 500;

/// Indentation spaces per level.
pub const INDENT_SPACES: usize = 2;

/// Arguments for the ls tool.
#[derive(Debug, Deserialize)]
pub struct LsArgs {
    /// Absolute path to directory.
    pub dir_path: String,
    /// Entry offset (1-indexed).
    #[serde(default = "default_offset")]
    pub offset: usize,
    /// Maximum entries to return.
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// Maximum directory depth.
    #[serde(default = "default_depth")]
    pub depth: usize,
}

fn default_offset() -> usize {
    DEFAULT_OFFSET
}

fn default_limit() -> usize {
    DEFAULT_LIMIT
}

fn default_depth() -> usize {
    DEFAULT_DEPTH
}

/// Directory entry with metadata.
#[derive(Debug)]
pub struct DirEntry {
    pub name: String,
    pub display_name: String,
    pub depth: usize,
    pub kind: EntryKind,
}

/// Type of directory entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    Directory,
    File,
    Symlink,
    Other,
}

impl From<FileType> for EntryKind {
    fn from(ft: FileType) -> Self {
        if ft.is_symlink() {
            EntryKind::Symlink
        } else if ft.is_dir() {
            EntryKind::Directory
        } else if ft.is_file() {
            EntryKind::File
        } else {
            EntryKind::Other
        }
    }
}

/// Truncate entry name if too long.
pub fn truncate_name(name: &str) -> String {
    if name.len() > MAX_ENTRY_LENGTH {
        name[..MAX_ENTRY_LENGTH].to_string()
    } else {
        name.to_string()
    }
}

/// Format a directory entry for display.
pub fn format_entry(entry: &DirEntry) -> String {
    let indent = " ".repeat(entry.depth * INDENT_SPACES);
    let suffix = match entry.kind {
        EntryKind::Directory => "/",
        EntryKind::Symlink => "@",
        EntryKind::Other => "?",
        EntryKind::File => "",
    };
    format!("{indent}{}{suffix}", entry.display_name)
}
