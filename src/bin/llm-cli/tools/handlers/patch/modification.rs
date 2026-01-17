//! Content modification logic for patches.

use crate::tools::error::ToolError;

/// Apply a modification to content based on context and remove patterns.
pub fn apply_modification(
    content: &str,
    context: Option<&str>,
    remove: Option<&str>,
    add: &str,
) -> Result<String, ToolError> {
    // If no context and no remove, append to file
    if context.is_none() && remove.is_none() {
        let mut result = content.to_string();
        if !result.is_empty() && !result.ends_with('\n') {
            result.push('\n');
        }
        result.push_str(add);
        return Ok(result);
    }

    let lines: Vec<&str> = content.lines().collect();
    let mut result_lines: Vec<String> = Vec::new();
    let mut i = 0;

    // Find the context location
    let context_pattern = context.map(|c| c.lines().collect::<Vec<_>>());
    let remove_pattern = remove.map(|r| r.lines().collect::<Vec<_>>());

    while i < lines.len() {
        // Check if we found the context
        let found_context = context_pattern.as_ref().is_none_or(|ctx_lines| {
            if i + ctx_lines.len() <= lines.len() {
                ctx_lines
                    .iter()
                    .zip(&lines[i..i + ctx_lines.len()])
                    .all(|(c, l)| l.contains(c.trim()))
            } else {
                false
            }
        });

        if found_context {
            // Add context lines
            if let Some(ctx_lines) = &context_pattern {
                for j in 0..ctx_lines.len() {
                    result_lines.push(lines[i + j].to_string());
                }
                i += ctx_lines.len();
            }

            // Skip remove lines
            if let Some(rem_lines) = &remove_pattern {
                for rem in rem_lines {
                    if i < lines.len() && lines[i].contains(rem.trim()) {
                        i += 1;
                    }
                }
            }

            // Add new lines
            for new_line in add.lines() {
                result_lines.push(new_line.to_string());
            }

            // Add remaining lines
            while i < lines.len() {
                result_lines.push(lines[i].to_string());
                i += 1;
            }

            return Ok(result_lines.join("\n"));
        } else {
            result_lines.push(lines[i].to_string());
            i += 1;
        }
    }

    // If context not found, append at end
    if !add.is_empty() {
        for new_line in add.lines() {
            result_lines.push(new_line.to_string());
        }
    }

    Ok(result_lines.join("\n"))
}
