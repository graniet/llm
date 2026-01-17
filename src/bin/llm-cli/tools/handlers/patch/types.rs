//! Patch data types.

#![allow(dead_code)]

/// A complete patch containing multiple hunks.
#[derive(Debug, Clone)]
pub struct Patch {
    pub hunks: Vec<PatchHunk>,
}

/// A single patch operation.
#[derive(Debug, Clone)]
pub enum PatchHunk {
    /// Add a new file.
    Add { path: String, content: String },
    /// Delete a file.
    Delete { path: String },
    /// Update an existing file.
    Update {
        path: String,
        new_path: Option<String>,
        context: Option<String>,
        remove: Option<String>,
        add: String,
    },
}

impl PatchHunk {
    /// Get the file path affected by this hunk.
    pub fn path(&self) -> &str {
        match self {
            Self::Add { path, .. } => path,
            Self::Delete { path } => path,
            Self::Update { path, .. } => path,
        }
    }

    /// Check if this is a mutating operation.
    pub const fn is_mutating(&self) -> bool {
        true // All patch operations are mutating
    }
}
