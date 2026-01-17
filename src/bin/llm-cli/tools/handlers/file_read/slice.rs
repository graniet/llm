//! Slice-based file reading.

use std::path::Path;

use crate::tools::error::ToolError;

use super::types::truncate_line;

/// Read a simple slice of lines from a file.
pub fn read_slice(path: &Path, offset: usize, limit: usize) -> Result<Vec<String>, ToolError> {
    let content = std::fs::read_to_string(path).map_err(|e| ToolError::Execution(e.to_string()))?;

    let lines: Vec<&str> = content.lines().collect();
    if offset > lines.len() {
        return Err(ToolError::RespondToModel(
            "offset exceeds file length".to_string(),
        ));
    }

    let start = offset - 1;
    let end = (start + limit).min(lines.len());

    Ok(lines[start..end]
        .iter()
        .enumerate()
        .map(|(i, line)| {
            let line_num = start + i + 1;
            let formatted = truncate_line(line);
            format!("L{line_num}: {formatted}")
        })
        .collect())
}
