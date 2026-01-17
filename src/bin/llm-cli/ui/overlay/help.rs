use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
use ratatui::Frame;

use super::super::theme::Theme;

pub fn render_help(frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
    // Clear the area first
    frame.render_widget(Clear, area);

    let lines = vec![
        help_line("Navigation", "j/k or arrows, g/G, Ctrl+u/d", theme),
        help_line("Input (simple)", "Enter send, Shift+Enter newline", theme),
        help_line("Input (vi)", "i insert, Esc normal, Enter newline", theme),
        help_line("Focus", "Tab toggle input/messages", theme),
        help_line(
            "Actions",
            "Ctrl+n new, Ctrl+f fork, Ctrl+s save, Ctrl+p provider",
            theme,
        ),
        help_line(
            "Messages",
            "Enter expand/pager, p pager, D diff, y copy, d delete",
            theme,
        ),
        help_line("History", "Ctrl+z backtrack", theme),
        help_line("Commands", "/ open command palette", theme),
        help_line("Search", "/ (vi mode), Esc to close", theme),
        help_line("Exit", "Ctrl+c", theme),
    ];

    let paragraph = Paragraph::new(Text::from(lines))
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(theme.border_focused)
                .style(Style::default().bg(Color::Black))
                .title(" Help "),
        )
        .wrap(Wrap { trim: true });

    frame.render_widget(paragraph, area);
}

fn help_line<'a>(key: &'a str, desc: &'a str, theme: &Theme) -> Line<'a> {
    Line::from(vec![
        Span::styled(
            format!("{}: ", key),
            Style::default().add_modifier(Modifier::BOLD),
        ),
        Span::styled(desc, theme.muted),
    ])
}
