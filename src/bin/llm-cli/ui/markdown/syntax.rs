use std::sync::OnceLock;

use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use syntect::easy::HighlightLines;
use syntect::highlighting::{Style as SyntectStyle, Theme, ThemeSet};
use syntect::parsing::SyntaxSet;

static SYNTAX_SET: OnceLock<SyntaxSet> = OnceLock::new();
static THEME: OnceLock<Theme> = OnceLock::new();

pub fn highlight_code(code: &str, language: Option<&str>) -> Vec<Line<'static>> {
    let syntax_set = syntax_set();
    let syntax = language
        .and_then(|lang| syntax_set.find_syntax_by_token(lang))
        .unwrap_or_else(|| syntax_set.find_syntax_plain_text());
    let theme = theme();
    let mut highlighter = HighlightLines::new(syntax, theme);
    code.lines()
        .map(|line| highlight_line(&mut highlighter, line))
        .collect()
}

fn highlight_line(highlighter: &mut HighlightLines<'_>, line: &str) -> Line<'static> {
    let regions = highlighter
        .highlight_line(line, syntax_set())
        .unwrap_or_default();
    let spans = regions
        .into_iter()
        .map(|(style, text)| Span::styled(text.to_string(), syntect_style(style)))
        .collect::<Vec<_>>();
    Line::from(spans)
}

fn syntect_style(style: SyntectStyle) -> Style {
    let fg = Color::Rgb(style.foreground.r, style.foreground.g, style.foreground.b);
    Style::default().fg(fg)
}

fn syntax_set() -> &'static SyntaxSet {
    SYNTAX_SET.get_or_init(SyntaxSet::load_defaults_newlines)
}

fn theme() -> &'static Theme {
    THEME.get_or_init(|| {
        let themes = ThemeSet::load_defaults();
        themes
            .themes
            .get("base16-ocean.dark")
            .cloned()
            .unwrap_or_else(|| themes.themes.values().next().cloned().unwrap_or_default())
    })
}
