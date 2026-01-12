use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

use crate::diff::{DiffHunk, DiffLine, LineKind};
use crate::runtime::DiffViewerState;

use super::super::theme::Theme;

const LINE_NUMBER_WIDTH: usize = 4;
const DIFF_HELP_HEIGHT: u16 = 1;

pub fn render_diff_viewer(
    frame: &mut Frame<'_>,
    area: Rect,
    state: &DiffViewerState,
    theme: &Theme,
) {
    let block = Block::default().borders(Borders::ALL).title("Diff Viewer");
    frame.render_widget(block.clone(), area);
    let inner = block.inner(area);
    if inner.height == 0 {
        return;
    }
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(DIFF_HELP_HEIGHT)])
        .split(inner);
    let text = build_diff_text(state, theme);
    let paragraph = Paragraph::new(text)
        .wrap(Wrap { trim: false })
        .scroll((state.scroll, 0));
    frame.render_widget(paragraph, chunks[0]);
    let help = Paragraph::new("y accept · n reject · s skip · A accept all · Enter apply · q quit")
        .style(theme.muted);
    frame.render_widget(help, chunks[1]);
}

fn build_diff_text(state: &DiffViewerState, theme: &Theme) -> Text<'static> {
    let mut lines = Vec::new();
    let total_files = state.diff.files.len().max(1);
    if let Some(file) = state.diff.files.get(state.file_index) {
        lines.push(Line::styled(
            format!(
                "File {} of {}: {}",
                state.file_index + 1,
                total_files,
                file.new_path
            ),
            theme.accent,
        ));
        if let Some(hunk) = file.hunks.get(state.hunk_index) {
            lines.extend(hunk_header_lines(hunk, theme));
            lines.extend(hunk_body_lines(hunk, theme));
        } else {
            lines.push(Line::styled("No hunks in file.", theme.muted));
        }
    } else {
        lines.push(Line::styled("No diff loaded.", theme.muted));
    }
    Text::from(lines)
}

fn hunk_header_lines(hunk: &DiffHunk, theme: &Theme) -> Vec<Line<'static>> {
    let decision = match hunk.decision {
        crate::diff::HunkDecision::Pending => "[ ]",
        crate::diff::HunkDecision::Accepted => "[A]",
        crate::diff::HunkDecision::Rejected => "[R]",
        crate::diff::HunkDecision::Skipped => "[S]",
    };
    vec![
        Line::styled(format!("{decision} {}", hunk.header), theme.diff_header),
        Line::styled("old new  | diff", theme.diff_header),
    ]
}

fn hunk_body_lines(hunk: &DiffHunk, theme: &Theme) -> Vec<Line<'static>> {
    let mut cursor = HunkCursor::new(hunk.old_start, hunk.new_start);
    hunk.lines
        .iter()
        .map(|line| cursor.render(line, theme))
        .collect()
}

struct HunkCursor {
    old_line: usize,
    new_line: usize,
}

impl HunkCursor {
    fn new(old_line: usize, new_line: usize) -> Self {
        Self { old_line, new_line }
    }

    fn render(&mut self, line: &DiffLine, theme: &Theme) -> Line<'static> {
        let (old_label, new_label, prefix, style) = self.describe(line, theme);
        let label = format!("{old_label} {new_label} | {prefix}");
        let spans = vec![
            Span::styled(label, theme.diff_lineno),
            Span::styled(line.content.clone(), style),
        ];
        Line::from(spans)
    }

    fn describe(
        &mut self,
        line: &DiffLine,
        theme: &Theme,
    ) -> (String, String, &'static str, ratatui::style::Style) {
        match line.kind {
            LineKind::Context => {
                let old = format!("{:>width$}", self.old_line, width = LINE_NUMBER_WIDTH);
                let new = format!("{:>width$}", self.new_line, width = LINE_NUMBER_WIDTH);
                self.old_line += 1;
                self.new_line += 1;
                (old, new, " ", theme.assistant)
            }
            LineKind::Remove => {
                let old = format!("{:>width$}", self.old_line, width = LINE_NUMBER_WIDTH);
                self.old_line += 1;
                (old, blank_number(), "-", theme.diff_remove)
            }
            LineKind::Add => {
                let new = format!("{:>width$}", self.new_line, width = LINE_NUMBER_WIDTH);
                self.new_line += 1;
                (blank_number(), new, "+", theme.diff_add)
            }
        }
    }
}

fn blank_number() -> String {
    " ".repeat(LINE_NUMBER_WIDTH)
}
