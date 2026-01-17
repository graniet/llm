//! Indentation-aware file reading.

use std::collections::VecDeque;
use std::path::Path;

use crate::tools::error::ToolError;

use super::types::{format_line, measure_indent, IndentationArgs, LineRecord, TAB_WIDTH};

/// Read with indentation-aware block extraction.
pub fn read_indentation(
    path: &Path,
    offset: usize,
    limit: usize,
    opts: IndentationArgs,
) -> Result<Vec<String>, ToolError> {
    let content = std::fs::read_to_string(path).map_err(|e| ToolError::Execution(e.to_string()))?;

    let lines: Vec<LineRecord> = content
        .lines()
        .enumerate()
        .map(|(i, line)| LineRecord {
            number: i + 1,
            raw: line.to_string(),
            indent: measure_indent(line),
        })
        .collect();

    if lines.is_empty() {
        return Ok(Vec::new());
    }

    let anchor_line = opts.anchor_line.unwrap_or(offset);
    if anchor_line == 0 || anchor_line > lines.len() {
        return Err(ToolError::RespondToModel(
            "anchor_line exceeds file length".to_string(),
        ));
    }

    let anchor_index = anchor_line - 1;
    let effective_indents = compute_effective_indents(&lines);
    let anchor_indent = effective_indents[anchor_index];

    let min_indent = if opts.max_levels == 0 {
        0
    } else {
        anchor_indent.saturating_sub(opts.max_levels * TAB_WIDTH)
    };

    let final_limit = limit.min(opts.max_lines.unwrap_or(limit)).min(lines.len());
    if final_limit == 1 {
        return Ok(vec![format_line(&lines[anchor_index])]);
    }

    let mut out: VecDeque<&LineRecord> = VecDeque::with_capacity(limit);
    out.push_back(&lines[anchor_index]);

    let mut i: isize = anchor_index as isize - 1;
    let mut j: usize = anchor_index + 1;

    while out.len() < final_limit {
        let mut progressed = false;

        // Expand upward
        if i >= 0 {
            let iu = i as usize;
            if effective_indents[iu] >= min_indent {
                out.push_front(&lines[iu]);
                progressed = true;
                i -= 1;
            } else {
                i = -1;
            }
        }

        // Expand downward
        if j < lines.len() && out.len() < final_limit {
            if effective_indents[j] >= min_indent {
                out.push_back(&lines[j]);
                progressed = true;
                j += 1;
            } else {
                j = lines.len();
            }
        }

        if !progressed {
            break;
        }
    }

    // Trim empty lines at boundaries
    while out.front().is_some_and(|l| l.raw.trim().is_empty()) {
        out.pop_front();
    }
    while out.back().is_some_and(|l| l.raw.trim().is_empty()) {
        out.pop_back();
    }

    Ok(out.iter().map(|r| format_line(r)).collect())
}

/// Compute effective indentation levels, carrying forward for blank lines.
fn compute_effective_indents(records: &[LineRecord]) -> Vec<usize> {
    let mut effective = Vec::with_capacity(records.len());
    let mut prev = 0;
    for record in records {
        if record.raw.trim().is_empty() {
            effective.push(prev);
        } else {
            prev = record.indent;
            effective.push(prev);
        }
    }
    effective
}
