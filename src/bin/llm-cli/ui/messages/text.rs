use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span, Text};

use crate::conversation::{
    ConversationMessage, MessageKind, MessageRole, ToolInvocation, ToolResult,
};

use super::super::markdown::render_markdown;
use super::super::theme::{borders, indicators, Theme};

/// Left border character for message styling
const BORDER_CHAR: &str = borders::LEFT_BORDER;

pub(super) fn render_message_text(
    message: &ConversationMessage,
    theme: &Theme,
    collapsed: bool,
) -> Text<'static> {
    // Check if this is a dialogue message with participant info
    let participant_header = build_participant_header(message);

    let mut rendered = match (&message.role, &message.kind) {
        (MessageRole::User, MessageKind::Text(content)) => {
            render_bordered_message(content, theme.user_border, theme.user_bg, theme)
        }
        (MessageRole::Assistant, MessageKind::Text(content)) => {
            // For dialogue messages, use participant color for border
            if let Some(color) = message.metadata.participant_color {
                let (r, g, b) = color.rgb();
                let border_style = Style::default().fg(Color::Rgb(r, g, b));
                render_bordered_message(content, border_style, theme.assistant_bg, theme)
            } else {
                render_bordered_message(content, theme.assistant_border, theme.assistant_bg, theme)
            }
        }
        (MessageRole::Tool, MessageKind::Text(content)) => {
            render_bordered_message(content, theme.tool_border, theme.tool_bg, theme)
        }
        (MessageRole::Error, MessageKind::Error(content)) | (_, MessageKind::Error(content)) => {
            render_error_message(content, theme)
        }
        (_, MessageKind::ToolCall(invocation)) => render_tool_call(invocation, theme),
        (_, MessageKind::ToolResult(result)) => render_tool_result(result, theme, collapsed),
        (_, MessageKind::Text(content)) => render_markdown(content),
    };

    // Prepend participant header if in dialogue mode
    if let Some(header) = participant_header {
        let mut lines = vec![header];
        lines.extend(rendered.lines);
        rendered = Text::from(lines);
    }

    rendered
}

/// Builds a participant header line if this message is part of a dialogue.
fn build_participant_header(message: &ConversationMessage) -> Option<Line<'static>> {
    let name = message.metadata.participant_name.as_ref()?;
    let color = message.metadata.participant_color?;
    let (r, g, b) = color.rgb();

    Some(Line::from(vec![
        Span::styled(
            format!("{BORDER_CHAR} "),
            Style::default().fg(Color::Rgb(r, g, b)),
        ),
        Span::styled(
            format!("[{}]", name),
            Style::default().fg(Color::Rgb(r, g, b)),
        ),
    ]))
}

/// Renders a message with a colored left border
fn render_bordered_message(
    content: &str,
    border_style: Style,
    bg_style: Style,
    theme: &Theme,
) -> Text<'static> {
    let markdown_text = render_markdown(content);
    let mut lines = Vec::with_capacity(markdown_text.lines.len());

    for line in markdown_text.lines {
        let bordered_line = add_left_border(line, border_style, bg_style);
        lines.push(bordered_line);
    }

    // Add empty line with border for visual separation if content is not empty
    if !lines.is_empty() {
        let empty_border = Line::from(vec![Span::styled(format!("{BORDER_CHAR} "), border_style)]);
        lines.push(empty_border);
    }

    let mut text = Text::from(lines);
    text.style = theme.assistant; // Base style
    text
}

/// Adds a colored left border to a line
fn add_left_border(line: Line<'static>, border_style: Style, bg_style: Style) -> Line<'static> {
    let mut spans = Vec::with_capacity(line.spans.len() + 1);

    // Add border character
    spans.push(Span::styled(format!("{BORDER_CHAR} "), border_style));

    // Add original spans with background if applicable
    for mut span in line.spans {
        if bg_style != Style::default() {
            span.style = span.style.patch(bg_style);
        }
        spans.push(span);
    }

    Line::from(spans)
}

/// Renders an error message with red border
fn render_error_message(content: &str, theme: &Theme) -> Text<'static> {
    let mut lines = Vec::new();

    // Error header
    lines.push(Line::from(vec![
        Span::styled(format!("{BORDER_CHAR} "), theme.error_border),
        Span::styled(format!("{} Error", indicators::CROSS), theme.error),
    ]));

    // Error content
    for line_content in content.lines() {
        lines.push(Line::from(vec![
            Span::styled(format!("{BORDER_CHAR} "), theme.error_border),
            Span::styled(line_content.to_string(), theme.error),
        ]));
    }

    Text::from(lines)
}

/// Renders a tool call in compact format
fn render_tool_call(invocation: &ToolInvocation, theme: &Theme) -> Text<'static> {
    let mut lines = Vec::new();

    // Compact single-line format: ● ToolName (arg1, arg2...)
    let args_preview = truncate_args(&invocation.arguments, 50);

    lines.push(Line::from(vec![
        Span::styled(format!("{BORDER_CHAR} "), theme.tool_border),
        Span::styled(format!("{} ", indicators::BULLET), theme.tool),
        Span::styled(invocation.name.clone(), theme.tool),
        Span::styled(format!(" {args_preview}"), theme.tool_dim),
    ]));

    Text::from(lines)
}

/// Renders a tool result in compact format
fn render_tool_result(result: &ToolResult, theme: &Theme, collapsed: bool) -> Text<'static> {
    let mut lines = Vec::new();

    // Status indicator
    let (status_icon, status_style) = if result.success {
        (indicators::CHECK, theme.status_ok)
    } else {
        (indicators::CROSS, theme.status_error)
    };

    // Expand/collapse indicator
    let expand_icon = if collapsed {
        indicators::EXPAND
    } else {
        indicators::COLLAPSE
    };

    // Header line: ▎ ▸ ✓ ToolName
    lines.push(Line::from(vec![
        Span::styled(format!("{BORDER_CHAR} "), theme.tool_border),
        Span::styled(format!("{expand_icon} "), theme.muted),
        Span::styled(format!("{status_icon} "), status_style),
        Span::styled(result.name.clone(), theme.tool),
    ]));

    if collapsed {
        // Collapsed: show summary
        let line_count = result.output.lines().count();
        lines.push(Line::from(vec![
            Span::styled(format!("{BORDER_CHAR} "), theme.tool_border),
            Span::styled(format!("  {line_count} lines"), theme.muted),
        ]));
    } else {
        // Expanded: show output with line numbers
        for (idx, line_content) in result.output.lines().enumerate() {
            let line_num = format!("{:>4} ", idx + 1);
            lines.push(Line::from(vec![
                Span::styled(format!("{BORDER_CHAR} "), theme.tool_border),
                Span::styled(line_num, theme.diff_lineno),
                Span::styled(line_content.to_string(), theme.tool_dim),
            ]));
        }
    }

    Text::from(lines)
}

/// Truncates and formats tool arguments for display
fn truncate_args(args: &str, max_len: usize) -> String {
    // Try to parse as JSON-like and extract key info
    let cleaned = args.trim();

    if cleaned.len() <= max_len {
        return cleaned.to_string();
    }

    // Truncate with ellipsis
    let truncated: String = cleaned.chars().take(max_len - 3).collect();
    format!("{truncated}...")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::{ConversationMessage, MessageKind, MessageRole, ToolInvocation};
    use crate::terminal::{ColorLevel, TerminalPalette};

    fn text_to_plain(text: &Text) -> String {
        text.lines
            .iter()
            .map(|line| {
                line.spans
                    .iter()
                    .map(|span| span.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn make_theme() -> Theme {
        let palette = TerminalPalette::new(ColorLevel::TrueColor);
        Theme::from_name("warm", &palette)
    }

    #[test]
    fn renders_tool_call_compact() {
        let theme = make_theme();
        let invocation = ToolInvocation {
            id: "call_1".to_string(),
            name: "read_file".to_string(),
            arguments: r#"{"path": "src/main.rs"}"#.to_string(),
            partial: false,
        };
        let message =
            ConversationMessage::new(MessageRole::Tool, MessageKind::ToolCall(invocation));
        let rendered = render_message_text(&message, &theme, false);
        let plain = text_to_plain(&rendered);

        // Should contain the tool name and be compact
        assert!(plain.contains("read_file"));
        assert!(plain.contains(indicators::BULLET));
    }

    #[test]
    fn renders_tool_result_collapsed() {
        let theme = make_theme();
        let result = ToolResult {
            id: "call_1".to_string(),
            name: "read_file".to_string(),
            output: "line1\nline2\nline3\nline4\nline5".to_string(),
            success: true,
        };
        let message = ConversationMessage::new(MessageRole::Tool, MessageKind::ToolResult(result));
        let rendered = render_message_text(&message, &theme, true);
        let plain = text_to_plain(&rendered);

        // Should show collapsed indicator and line count
        assert!(plain.contains(indicators::EXPAND));
        assert!(plain.contains("5 lines"));
    }

    #[test]
    fn renders_tool_result_expanded() {
        let theme = make_theme();
        let result = ToolResult {
            id: "call_1".to_string(),
            name: "read_file".to_string(),
            output: "content line".to_string(),
            success: true,
        };
        let message = ConversationMessage::new(MessageRole::Tool, MessageKind::ToolResult(result));
        let rendered = render_message_text(&message, &theme, false);
        let plain = text_to_plain(&rendered);

        // Should show collapse indicator and content
        assert!(plain.contains(indicators::COLLAPSE));
        assert!(plain.contains("content line"));
    }

    #[test]
    fn renders_user_message_with_border() {
        let theme = make_theme();
        let message = ConversationMessage::new(
            MessageRole::User,
            MessageKind::Text("Hello world".to_string()),
        );
        let rendered = render_message_text(&message, &theme, false);
        let plain = text_to_plain(&rendered);

        // Should contain border character
        assert!(plain.contains(BORDER_CHAR));
        assert!(plain.contains("Hello world"));
    }

    #[test]
    fn truncate_args_short() {
        let short = r#"{"a": 1}"#;
        assert_eq!(truncate_args(short, 50), short);
    }

    #[test]
    fn truncate_args_long() {
        let long = "a".repeat(100);
        let truncated = truncate_args(&long, 50);
        assert!(truncated.len() <= 50);
        assert!(truncated.ends_with("..."));
    }
}
