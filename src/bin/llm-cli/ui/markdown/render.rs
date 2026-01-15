use pulldown_cmark::{CodeBlockKind, Event, Options, Parser, Tag, TagEnd};
use ratatui::style::Style;
use ratatui::text::{Line, Span, Text};

use super::styles::MarkdownStyles;
use super::syntax::highlight_code;

/// Code block visual configuration
mod code_block {
    pub const TOP_LEFT: &str = "┌";
    pub const TOP_RIGHT: &str = "┐";
    pub const BOTTOM_LEFT: &str = "└";
    pub const BOTTOM_RIGHT: &str = "┘";
    pub const HORIZONTAL: &str = "─";
    pub const VERTICAL: &str = "│";
}

pub fn render_markdown(input: &str) -> Text<'static> {
    let mut renderer = Renderer::new();
    renderer.run(input);
    Text::from(renderer.lines)
}

struct Renderer {
    lines: Vec<Line<'static>>,
    current: Vec<Span<'static>>,
    styles: MarkdownStyles,
    inline_styles: Vec<Style>,
    list_stack: Vec<Option<u64>>,
    pending_marker: Option<Vec<Span<'static>>>,
    in_code_block: bool,
    code_block_lang: Option<String>,
    code_block_buf: String,
}

impl Renderer {
    fn new() -> Self {
        Self {
            lines: Vec::new(),
            current: Vec::new(),
            styles: MarkdownStyles::default(),
            inline_styles: Vec::new(),
            list_stack: Vec::new(),
            pending_marker: None,
            in_code_block: false,
            code_block_lang: None,
            code_block_buf: String::new(),
        }
    }

    fn run(&mut self, input: &str) {
        let mut options = Options::empty();
        options.insert(Options::ENABLE_STRIKETHROUGH);
        let parser = Parser::new_ext(input, options);
        for event in parser {
            self.handle_event(event);
        }
        self.flush_line();
    }

    fn handle_event(&mut self, event: Event<'_>) {
        match event {
            Event::Start(tag) => self.start_tag(tag),
            Event::End(tag) => self.end_tag(tag),
            Event::Text(text) => self.push_text(&text),
            Event::Code(code) => self.push_inline_code(&code),
            Event::SoftBreak => self.soft_break(),
            Event::HardBreak => self.hard_break(),
            Event::Rule => self.rule(),
            _ => {}
        }
    }

    fn start_tag(&mut self, tag: Tag<'_>) {
        match tag {
            Tag::Heading { level, .. } => self.push_heading_style(level),
            Tag::Emphasis => self.inline_styles.push(self.styles.emphasis),
            Tag::Strong => self.inline_styles.push(self.styles.strong),
            Tag::Strikethrough => self.inline_styles.push(self.styles.strikethrough),
            Tag::BlockQuote(_) => self.inline_styles.push(self.styles.blockquote),
            Tag::List(start) => self.list_stack.push(start),
            Tag::Item => self.push_list_marker(),
            Tag::CodeBlock(kind) => self.start_code_block(kind),
            Tag::Link { .. } => self.inline_styles.push(self.styles.link),
            _ => {}
        }
    }

    fn end_tag(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Heading(_) => self.flush_line(),
            TagEnd::Emphasis
            | TagEnd::Strong
            | TagEnd::Strikethrough
            | TagEnd::BlockQuote(_)
            | TagEnd::Link => {
                self.inline_styles.pop();
            }
            TagEnd::List(_) => {
                self.list_stack.pop();
                self.flush_line();
            }
            TagEnd::Item => self.flush_line(),
            TagEnd::CodeBlock => self.end_code_block(),
            _ => {}
        }
    }

    fn push_text(&mut self, text: &str) {
        if self.in_code_block {
            self.code_block_buf.push_str(text);
            return;
        }
        self.current.push(self.styled_span(text));
    }

    fn push_inline_code(&mut self, code: &str) {
        // Inline code with distinct styling
        let styled = format!(" {code} ");
        self.current.push(Span::styled(styled, self.styles.code));
    }

    fn push_heading_style(&mut self, level: pulldown_cmark::HeadingLevel) {
        let style = match level {
            pulldown_cmark::HeadingLevel::H1 => self.styles.h1,
            pulldown_cmark::HeadingLevel::H2 => self.styles.h2,
            _ => self.styles.h3,
        };
        self.inline_styles.push(style);
    }

    fn push_list_marker(&mut self) {
        let marker = match self.list_stack.last().copied().flatten() {
            Some(num) => {
                let value = format!("{}. ", num);
                Span::styled(value, self.styles.list_marker)
            }
            None => Span::styled("• ".to_string(), self.styles.list_marker),
        };
        self.pending_marker = Some(vec![marker]);
    }

    fn start_code_block(&mut self, kind: CodeBlockKind<'_>) {
        self.flush_line();
        self.in_code_block = true;
        self.code_block_buf.clear();
        self.code_block_lang = match kind {
            CodeBlockKind::Fenced(lang) => {
                let lang_str = lang.to_string();
                if lang_str.is_empty() {
                    None
                } else {
                    Some(lang_str)
                }
            }
            CodeBlockKind::Indented => None,
        };
    }

    fn end_code_block(&mut self) {
        let lang = self.code_block_lang.take();
        let code_lines = highlight_code(&self.code_block_buf, lang.as_deref());

        // Build bordered code block
        self.render_code_block_with_border(code_lines, lang.as_deref());

        self.in_code_block = false;
        self.code_block_buf.clear();
    }

    fn render_code_block_with_border(
        &mut self,
        code_lines: Vec<Line<'static>>,
        lang: Option<&str>,
    ) {
        let header_style = self.styles.code_header;
        let border_style = self.styles.code_border;
        let bg_style = self.styles.code_bg;

        // Calculate width for the border
        let content_width = code_lines
            .iter()
            .map(|l| line_display_width(l))
            .max()
            .unwrap_or(20)
            .max(20);

        let box_width = content_width + 4; // 2 for "│ " prefix, 2 for " │" suffix

        // Top border with language header
        let header_text = lang.unwrap_or("code");
        let header_line = build_header_line(header_text, box_width, header_style, border_style);
        self.lines.push(header_line);

        // Code lines with vertical borders
        for code_line in code_lines {
            let bordered = build_code_line(code_line, border_style, bg_style);
            self.lines.push(bordered);
        }

        // Bottom border
        let bottom_line = build_bottom_line(box_width, border_style);
        self.lines.push(bottom_line);

        // Empty line after code block for spacing
        self.lines.push(Line::default());
    }

    fn soft_break(&mut self) {
        self.current.push(self.styled_span(" "));
    }

    fn hard_break(&mut self) {
        self.flush_line();
    }

    fn rule(&mut self) {
        self.flush_line();
        self.lines.push(Line::from("─".repeat(24)));
    }

    fn flush_line(&mut self) {
        if self.current.is_empty() && self.pending_marker.is_none() {
            return;
        }
        let mut spans = Vec::new();
        if let Some(marker) = self.pending_marker.take() {
            spans.extend(marker);
        }
        spans.append(&mut self.current);
        self.lines.push(Line::from(spans));
    }

    fn styled_span(&self, text: &str) -> Span<'static> {
        let style = self.inline_styles.last().copied().unwrap_or_default();
        Span::styled(text.to_string(), style)
    }
}

/// Builds the header line for a code block: ┌─ language ─────┐
fn build_header_line(
    lang: &str,
    width: usize,
    header_style: Style,
    border_style: Style,
) -> Line<'static> {
    let lang_display = format!(" {} ", lang);
    let lang_width = lang_display.chars().count();
    let remaining = width.saturating_sub(2 + lang_width); // 2 for corners
    let right_border = code_block::HORIZONTAL.repeat(remaining.saturating_sub(1));

    Line::from(vec![
        Span::styled(code_block::TOP_LEFT.to_string(), border_style),
        Span::styled(code_block::HORIZONTAL.to_string(), border_style),
        Span::styled(lang_display, header_style),
        Span::styled(right_border, border_style),
        Span::styled(code_block::TOP_RIGHT.to_string(), border_style),
    ])
}

/// Builds a code line with vertical borders: │ code │
fn build_code_line(
    code_line: Line<'static>,
    border_style: Style,
    bg_style: Style,
) -> Line<'static> {
    let mut spans = Vec::with_capacity(code_line.spans.len() + 3);

    // Left border
    spans.push(Span::styled(
        format!("{} ", code_block::VERTICAL),
        border_style,
    ));

    // Code content with background
    for mut span in code_line.spans {
        if bg_style != Style::default() {
            span.style = span.style.patch(bg_style);
        }
        spans.push(span);
    }

    // Right border (we don't add it to avoid line length issues)
    // Terminal will handle the background extending

    Line::from(spans)
}

/// Builds the bottom border: └──────────────────┘
fn build_bottom_line(width: usize, border_style: Style) -> Line<'static> {
    let middle_width = width.saturating_sub(2); // 2 for corners
    let middle = code_block::HORIZONTAL.repeat(middle_width);

    Line::from(vec![
        Span::styled(code_block::BOTTOM_LEFT.to_string(), border_style),
        Span::styled(middle, border_style),
        Span::styled(code_block::BOTTOM_RIGHT.to_string(), border_style),
    ])
}

/// Calculates the display width of a line
fn line_display_width(line: &Line<'_>) -> usize {
    line.spans
        .iter()
        .map(|span| span.content.chars().count())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_plain_text() {
        let text = render_markdown("Hello world");
        assert_eq!(text.lines.len(), 1);
    }

    #[test]
    fn renders_code_block_with_border() {
        let input = "```rust\nfn main() {}\n```";
        let text = render_markdown(input);

        // Should have header, code line, bottom border, and empty line
        assert!(text.lines.len() >= 3);

        // First line should be header with language
        let first_line = &text.lines[0];
        let first_content: String = first_line
            .spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect();
        assert!(first_content.contains("rust"));
    }

    #[test]
    fn renders_inline_code() {
        let text = render_markdown("Use `println!` macro");
        let content: String = text.lines[0]
            .spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect();
        assert!(content.contains("println!"));
    }

    #[test]
    fn renders_list_with_markers() {
        let input = "- item 1\n- item 2";
        let text = render_markdown(input);

        let content: String = text
            .lines
            .iter()
            .flat_map(|l| l.spans.iter().map(|s| s.content.as_ref()))
            .collect();
        assert!(content.contains("•"));
    }
}
