use super::theme::{indicators, Theme};
use crate::runtime::{usage_for, AppState, AppStatus, ContextUsage};
use crate::terminal::{AnimationLevel, Rgb, TerminalPalette};
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};
use ratatui::Frame;
use unicode_width::UnicodeWidthStr;

const CONTEXT_MIN_WIDTH: u16 = 60;
const MODE_MIN_WIDTH: u16 = 80;
const TOKENS_MIN_WIDTH: u16 = 100;
const CONTEXT_WARN_PCT: f32 = 80.0;
const CONTEXT_CAUTION_PCT: f32 = 60.0;

pub struct StatusLine {
    pub left: Vec<Span<'static>>,
    pub right: Vec<Span<'static>>,
}

impl StatusLine {
    pub fn new(left: Vec<Span<'static>>, right: Vec<Span<'static>>) -> Self {
        Self { left, right }
    }
}

pub fn render_status(frame: &mut Frame<'_>, area: Rect, line: StatusLine, theme: &Theme) {
    let left_width = spans_width(&line.left);
    let right_width = spans_width(&line.right);
    let filler_width = area.width.saturating_sub(left_width + right_width);
    let filler = Span::styled(" ".repeat(filler_width as usize), theme.status);
    let mut spans = Vec::with_capacity(line.left.len() + line.right.len() + 1);
    spans.extend(line.left);
    spans.push(filler);
    spans.extend(line.right);
    let paragraph = Paragraph::new(Line::from(spans)).wrap(Wrap { trim: true });
    frame.render_widget(paragraph, area);
}

fn spans_width(spans: &[Span<'static>]) -> u16 {
    spans.iter().map(|span| span.content.width() as u16).sum()
}

pub fn build_status_line(state: &AppState, theme: &Theme) -> StatusLine {
    let width = state.terminal_size.0;
    let left = build_left_spans(state, theme, width);
    let right = build_right_spans(state, theme, width);
    StatusLine::new(left, right)
}

fn build_left_spans(state: &AppState, theme: &Theme, width: u16) -> Vec<Span<'static>> {
    let (provider, model) = state
        .active_conversation()
        .map(|conv| {
            (
                conv.provider_id.to_string(),
                conv.model.clone().unwrap_or_else(|| "default".to_string()),
            )
        })
        .unwrap_or_else(|| ("-".to_string(), "-".to_string()));
    let label = if width < CONTEXT_MIN_WIDTH {
        provider
    } else {
        format!("{provider} · {model}")
    };
    vec![Span::styled(label, theme.status)]
}

fn build_right_spans(state: &AppState, theme: &Theme, width: u16) -> Vec<Span<'static>> {
    let mut spans = Vec::new();

    if width >= CONTEXT_MIN_WIDTH {
        if let Some(usage) = state
            .active_conversation()
            .map(|conv| usage_for(conv, &state.config))
        {
            push_context_spans(&mut spans, usage, theme, width);
        }
    }

    if width >= MODE_MIN_WIDTH {
        push_mode_spans(&mut spans, state, theme);
    }

    let status_spans = build_status_spans(state, theme);
    if !status_spans.is_empty() {
        if !spans.is_empty() {
            spans.push(Span::styled(" · ", theme.status));
        }
        spans.extend(status_spans);
    }

    spans
}

fn push_context_spans(
    spans: &mut Vec<Span<'static>>,
    usage: ContextUsage,
    theme: &Theme,
    width: u16,
) {
    let percent = usage.percent().round() as u32;
    let style = context_style(usage.percent(), theme);
    spans.push(Span::styled(format!("{percent}% context"), style));

    if width >= TOKENS_MIN_WIDTH {
        spans.push(Span::styled(
            format!(" · {}/{} tokens", usage.used_tokens, usage.max_tokens),
            theme.status,
        ));
        if usage.percent() >= CONTEXT_WARN_PCT {
            spans.push(Span::styled(" · Context filling up", theme.status_warn));
        }
    }
}

fn push_mode_spans(spans: &mut Vec<Span<'static>>, state: &AppState, theme: &Theme) {
    let label = match state.config.ui.navigation_mode {
        crate::config::NavigationMode::Simple => "simple",
        crate::config::NavigationMode::Vi => "vi",
    };
    if !spans.is_empty() {
        spans.push(Span::styled(" · ", theme.status));
    }
    spans.push(Span::styled(format!("mode {label}"), theme.status));
}

fn context_style(percent: f32, theme: &Theme) -> ratatui::style::Style {
    if percent >= CONTEXT_WARN_PCT {
        theme.status_error
    } else if percent >= CONTEXT_CAUTION_PCT {
        theme.status_warn
    } else {
        theme.status_ok
    }
}

fn build_status_spans(state: &AppState, theme: &Theme) -> Vec<Span<'static>> {
    match &state.status {
        AppStatus::Idle => idle_spans(theme),
        AppStatus::Thinking => thinking_spans(state, theme),
        AppStatus::Streaming => streaming_spans(state, theme),
        AppStatus::Error(err) => error_spans(err, theme),
    }
}

/// Idle state: gray bullet with "idle" text
fn idle_spans(theme: &Theme) -> Vec<Span<'static>> {
    vec![
        Span::styled(format!("{} ", indicators::BULLET), theme.status),
        Span::styled("idle".to_string(), theme.status),
    ]
}

/// Thinking state: animated/pulsing indicator with status text
fn thinking_spans(state: &AppState, theme: &Theme) -> Vec<Span<'static>> {
    let mut spans = build_dynamic_indicator(state, theme, "Thinking...");
    append_elapsed(&mut spans, state, theme);
    spans
}

/// Streaming state: active indicator with token count
fn streaming_spans(state: &AppState, theme: &Theme) -> Vec<Span<'static>> {
    let mut spans = build_dynamic_indicator(state, theme, "Streaming");

    if let Some(tokens) = state.status_metrics.tokens() {
        spans.push(Span::styled(format!(" · {tokens} tok"), theme.status));
    }

    append_elapsed(&mut spans, state, theme);
    spans
}

/// Error state: red cross with error message
fn error_spans(err: &str, theme: &Theme) -> Vec<Span<'static>> {
    vec![
        Span::styled(format!("{} ", indicators::CROSS), theme.status_error),
        Span::styled(format!("error: {err}"), theme.error),
    ]
}

/// Builds a dynamic status indicator based on terminal capabilities
fn build_dynamic_indicator(state: &AppState, theme: &Theme, label: &str) -> Vec<Span<'static>> {
    let palette = TerminalPalette::new(state.terminal_caps.color_level);

    match state.animation.level(&state.terminal_caps) {
        AnimationLevel::Shimmer => shimmer_indicator(label, state.animation.frame(), &palette),
        AnimationLevel::Spinner => spinner_indicator(label, state.animation.frame(), theme),
        AnimationLevel::Static => static_indicator(label, theme),
    }
}

/// Shimmer animation: color gradient moving across text
fn shimmer_indicator(label: &str, frame: u64, palette: &TerminalPalette) -> Vec<Span<'static>> {
    // Start with animated bullet
    let bullet_color = palette.blend(
        Rgb::new(217, 119, 87),  // Orange base
        Rgb::new(255, 180, 140), // Lighter orange highlight
        pulse_intensity(frame),
    );

    let mut spans = vec![Span::styled(
        format!("{} ", indicators::BULLET),
        ratatui::style::Style::default().fg(bullet_color),
    )];

    // Shimmer effect on text
    let chars: Vec<char> = label.chars().collect();
    let len = chars.len().max(1);
    let center = (frame as usize) % len;
    let base = Rgb::new(140, 135, 130); // Muted gray
    let highlight = Rgb::new(217, 119, 87); // Orange highlight

    for (idx, ch) in chars.into_iter().enumerate() {
        let dist = idx.abs_diff(center);
        let t = match dist {
            0 => 1.0,
            1 => 0.6,
            2 => 0.3,
            _ => 0.1,
        };
        let color = palette.blend(base, highlight, t);
        spans.push(Span::styled(
            ch.to_string(),
            ratatui::style::Style::default().fg(color),
        ));
    }

    spans
}

/// Spinner animation: rotating bullet with status text
fn spinner_indicator(label: &str, frame: u64, theme: &Theme) -> Vec<Span<'static>> {
    let spinner = spinner_frame(frame);
    vec![
        Span::styled(format!("{spinner} "), theme.status_indicator),
        Span::styled(label.to_string(), theme.status),
    ]
}

/// Static indicator: simple bullet with text (for limited terminals)
fn static_indicator(label: &str, theme: &Theme) -> Vec<Span<'static>> {
    vec![
        Span::styled(format!("{} ", indicators::BULLET), theme.status_indicator),
        Span::styled(label.to_string(), theme.status),
    ]
}

fn append_elapsed(spans: &mut Vec<Span<'static>>, state: &AppState, theme: &Theme) {
    let Some(ms) = state.status_metrics.elapsed_ms() else {
        return;
    };
    let elapsed = format_elapsed(ms);
    spans.push(Span::styled(format!(" · {elapsed}"), theme.status));
}

fn format_elapsed(ms: u128) -> String {
    let secs = ms as f32 / 1000.0;
    format!("{secs:.1}s")
}

fn spinner_frame(frame: u64) -> &'static str {
    const FRAMES: [&str; 4] = ["◐", "◓", "◑", "◒"];
    let idx = (frame % FRAMES.len() as u64) as usize;
    FRAMES[idx]
}

/// Calculates pulse intensity for breathing effect (0.0 to 1.0)
fn pulse_intensity(frame: u64) -> f32 {
    let cycle = (frame % 20) as f32 / 20.0;
    let intensity = (cycle * std::f32::consts::PI * 2.0).sin();
    (intensity + 1.0) / 2.0 // Normalize to 0.0-1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pulse_intensity_range() {
        for frame in 0..100 {
            let intensity = pulse_intensity(frame);
            assert!((0.0..=1.0).contains(&intensity));
        }
    }

    #[test]
    fn spinner_cycles() {
        let frames: Vec<&str> = (0..8).map(spinner_frame).collect();
        // Should cycle through all 4 frames twice
        assert_eq!(frames[0], frames[4]);
        assert_eq!(frames[1], frames[5]);
    }

    #[test]
    fn format_elapsed_formats_correctly() {
        assert_eq!(format_elapsed(1500), "1.5s");
        assert_eq!(format_elapsed(10000), "10.0s");
    }
}
