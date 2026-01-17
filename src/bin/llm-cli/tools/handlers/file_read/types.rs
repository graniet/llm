//! Types for the file_read tool.

use serde::Deserialize;

/// Tab width for indentation calculation.
pub const TAB_WIDTH: usize = 4;

/// Maximum line length before truncation.
pub const MAX_LINE_LENGTH: usize = 500;

/// Default offset (1-indexed).
pub const DEFAULT_OFFSET: usize = 1;

/// Default limit.
pub const DEFAULT_LIMIT: usize = 2000;

/// Arguments for the file_read tool.
#[derive(Debug, Deserialize)]
pub struct FileReadArgs {
    pub file_path: String,
    #[serde(default = "default_offset")]
    pub offset: usize,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub mode: ReadMode,
    #[serde(default)]
    pub indentation: Option<IndentationArgs>,
}

fn default_offset() -> usize {
    DEFAULT_OFFSET
}

fn default_limit() -> usize {
    DEFAULT_LIMIT
}

#[derive(Debug, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ReadMode {
    #[default]
    Slice,
    Indentation,
}

#[derive(Debug, Deserialize, Clone)]
pub struct IndentationArgs {
    #[serde(default)]
    pub anchor_line: Option<usize>,
    #[serde(default)]
    pub max_levels: usize,
    #[serde(default)]
    pub include_siblings: bool,
    #[serde(default = "default_true")]
    pub include_header: bool,
    #[serde(default)]
    pub max_lines: Option<usize>,
}

fn default_true() -> bool {
    true
}

impl Default for IndentationArgs {
    fn default() -> Self {
        Self {
            anchor_line: None,
            max_levels: 0,
            include_siblings: false,
            include_header: true,
            max_lines: None,
        }
    }
}

/// Record for a single line with metadata.
#[derive(Debug)]
pub struct LineRecord {
    pub number: usize,
    pub raw: String,
    pub indent: usize,
}

/// Measure the indentation level of a line.
pub fn measure_indent(line: &str) -> usize {
    line.chars()
        .take_while(|c| matches!(c, ' ' | '\t'))
        .map(|c| if c == '\t' { TAB_WIDTH } else { 1 })
        .sum()
}

/// Truncate a line if it exceeds the maximum length.
pub fn truncate_line(line: &str) -> String {
    if line.len() > MAX_LINE_LENGTH {
        line[..MAX_LINE_LENGTH].to_string()
    } else {
        line.to_string()
    }
}

/// Format a line record for output.
pub fn format_line(record: &LineRecord) -> String {
    let formatted = truncate_line(&record.raw);
    format!("L{}: {formatted}", record.number)
}
