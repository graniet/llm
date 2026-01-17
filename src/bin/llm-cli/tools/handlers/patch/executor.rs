//! Patch executor that applies file modifications.

use std::fs;
use std::path::Path;

use super::modification::apply_modification;
use super::types::{Patch, PatchHunk};
use crate::tools::context::ToolContext;
use crate::tools::error::ToolError;

/// Apply a patch to the filesystem.
pub fn apply_patch(patch: &Patch, ctx: &ToolContext) -> Result<String, ToolError> {
    let mut results = Vec::new();

    for hunk in &patch.hunks {
        let result = apply_hunk(hunk, ctx)?;
        results.push(result);
    }

    Ok(results.join("\n"))
}

fn apply_hunk(hunk: &PatchHunk, ctx: &ToolContext) -> Result<String, ToolError> {
    match hunk {
        PatchHunk::Add { path, content } => apply_add(path, content, ctx),
        PatchHunk::Delete { path } => apply_delete(path, ctx),
        PatchHunk::Update {
            path,
            new_path,
            context,
            remove,
            add,
        } => apply_update(
            path,
            new_path.as_deref(),
            context.as_deref(),
            remove.as_deref(),
            add,
            ctx,
        ),
    }
}

fn apply_add(path: &str, content: &str, ctx: &ToolContext) -> Result<String, ToolError> {
    let full_path = resolve_path(path, ctx)?;

    // Check sandbox permissions
    if !ctx.is_write_allowed(full_path.to_string_lossy().as_ref()) {
        return Err(ToolError::Denied(format!(
            "Write not allowed to: {}",
            full_path.display()
        )));
    }

    // Create parent directories if needed
    if let Some(parent) = full_path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| ToolError::Execution(format!("Failed to create directory: {e}")))?;
    }

    // Write the file
    fs::write(&full_path, content)
        .map_err(|e| ToolError::Execution(format!("Failed to write file: {e}")))?;

    Ok(format!("Added file: {}", full_path.display()))
}

fn apply_delete(path: &str, ctx: &ToolContext) -> Result<String, ToolError> {
    let full_path = resolve_path(path, ctx)?;

    // Check sandbox permissions
    if !ctx.is_write_allowed(full_path.to_string_lossy().as_ref()) {
        return Err(ToolError::Denied(format!(
            "Delete not allowed for: {}",
            full_path.display()
        )));
    }

    // Check if file exists
    if !full_path.exists() {
        return Err(ToolError::RespondToModel(format!(
            "File does not exist: {}",
            full_path.display()
        )));
    }

    // Delete the file
    fs::remove_file(&full_path)
        .map_err(|e| ToolError::Execution(format!("Failed to delete file: {e}")))?;

    Ok(format!("Deleted file: {}", full_path.display()))
}

fn apply_update(
    path: &str,
    new_path: Option<&str>,
    context: Option<&str>,
    remove: Option<&str>,
    add: &str,
    ctx: &ToolContext,
) -> Result<String, ToolError> {
    let full_path = resolve_path(path, ctx)?;

    // Check sandbox permissions
    if !ctx.is_write_allowed(full_path.to_string_lossy().as_ref()) {
        return Err(ToolError::Denied(format!(
            "Write not allowed to: {}",
            full_path.display()
        )));
    }

    // Read current content
    let current_content = fs::read_to_string(&full_path)
        .map_err(|e| ToolError::Execution(format!("Failed to read file: {e}")))?;

    // Apply the modification
    let new_content = apply_modification(&current_content, context, remove, add)?;

    // Handle rename/move
    let target_path = if let Some(np) = new_path {
        let target = resolve_path(np, ctx)?;
        if !ctx.is_write_allowed(target.to_string_lossy().as_ref()) {
            return Err(ToolError::Denied(format!(
                "Write not allowed to: {}",
                target.display()
            )));
        }
        // Create parent directories if needed
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| ToolError::Execution(format!("Failed to create directory: {e}")))?;
        }
        // Delete old file
        fs::remove_file(&full_path)
            .map_err(|e| ToolError::Execution(format!("Failed to remove old file: {e}")))?;
        target
    } else {
        full_path.clone()
    };

    // Write the updated content
    fs::write(&target_path, new_content)
        .map_err(|e| ToolError::Execution(format!("Failed to write file: {e}")))?;

    if new_path.is_some() {
        Ok(format!(
            "Updated and moved: {} -> {}",
            full_path.display(),
            target_path.display()
        ))
    } else {
        Ok(format!("Updated file: {}", target_path.display()))
    }
}

fn resolve_path(path: &str, ctx: &ToolContext) -> Result<std::path::PathBuf, ToolError> {
    let path = Path::new(path);

    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(Path::new(&ctx.working_dir).join(path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_apply_add() {
        let dir = tempdir().unwrap();
        let ctx = ToolContext::new(dir.path().to_string_lossy().to_string());

        let result = apply_add("test.txt", "hello world", &ctx);
        assert!(result.is_ok());

        let content = fs::read_to_string(dir.path().join("test.txt")).unwrap();
        assert_eq!(content, "hello world");
    }

    #[test]
    fn test_apply_modification_append() {
        let content = "line 1\nline 2";
        let result = apply_modification(content, None, None, "line 3").unwrap();
        assert!(result.contains("line 3"));
    }
}
