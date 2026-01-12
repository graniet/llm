use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::text::Text;
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;

use crate::runtime::{AppState, Focus, InputMode, OverlayState};

use super::consts::{MIN_INPUT_HEIGHT, OVERLAY_HEIGHT_PCT, OVERLAY_WIDTH_PCT, STATUS_HEIGHT};
use super::input::{input_height, render_input, InputProps, ViMode};
use super::messages::{render_messages, MessageRenderProps, MessageRenderer};
use super::overlay::render_overlay;
use super::status::{build_status_line, render_status};
use super::theme::Theme;
use crate::terminal::TerminalPalette;

const MIN_WIDTH: u16 = 40;
const MIN_HEIGHT: u16 = 10;
const COMPACT_WIDTH: u16 = 60;
const OVERLAY_COMPACT_WIDTH: u16 = 90;
const OVERLAY_COMPACT_HEIGHT: u16 = 80;
const OVERLAY_MEDIUM_WIDTH: u16 = 70;
const OVERLAY_MEDIUM_HEIGHT: u16 = 70;

pub fn render_app(frame: &mut Frame<'_>, state: &AppState, renderer: &mut MessageRenderer) {
    let palette = TerminalPalette::new(state.terminal_caps.color_level);
    let theme = Theme::from_name(&state.config.ui.theme, &palette);
    let size = frame.area();
    if size.width < MIN_WIDTH || size.height < MIN_HEIGHT {
        render_too_small(frame, size, &theme);
        return;
    }
    let areas = split_rows(size, &state.input);
    let mut ctx = RenderContext {
        state,
        renderer,
        theme: &theme,
    };
    render_main(frame, &mut ctx, areas);
    let (overlay_w, overlay_h) = overlay_size(size.width);
    render_overlay(
        frame,
        centered_rect(overlay_w, overlay_h, size),
        &state.overlay,
        &theme,
    );
}

fn split_rows(area: Rect, input: &crate::input::InputBuffer) -> MainAreas {
    let input_h = input_height(input, area.width).max(MIN_INPUT_HEIGHT);
    let status_h = STATUS_HEIGHT;
    let main_h = area.height.saturating_sub(input_h + status_h);
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(main_h),
            Constraint::Length(status_h),
            Constraint::Length(input_h),
        ])
        .split(area);
    MainAreas {
        messages: rows[0],
        status: rows[1],
        input: rows[2],
    }
}

struct MainAreas {
    messages: Rect,
    status: Rect,
    input: Rect,
}

struct RenderContext<'a> {
    state: &'a AppState,
    renderer: &'a mut MessageRenderer,
    theme: &'a Theme,
}

fn render_main(frame: &mut Frame<'_>, ctx: &mut RenderContext<'_>, areas: MainAreas) {
    render_messages_area(frame, ctx, areas.messages);
    render_status(
        frame,
        areas.status,
        build_status_line(ctx.state, ctx.theme),
        ctx.theme,
    );
    render_input_area(frame, ctx, areas.input);
}

fn render_messages_area(frame: &mut Frame<'_>, ctx: &mut RenderContext<'_>, area: Rect) {
    let Some(conv) = ctx.state.active_conversation() else {
        return;
    };
    render_messages(
        frame,
        ctx.renderer,
        MessageRenderProps {
            area,
            messages: &conv.messages,
            theme: ctx.theme,
            scroll: ctx.state.scroll,
            selected: ctx.state.selected_message,
            collapse: &ctx.state.collapsible,
        },
    );
}

fn render_input_area(frame: &mut Frame<'_>, ctx: &mut RenderContext<'_>, area: Rect) {
    let placeholder = if area.width < COMPACT_WIDTH {
        "Message..."
    } else {
        "Type a message..."
    };

    // Convert InputMode to ViMode
    let vi_mode = match ctx.state.input_mode {
        InputMode::Insert => ViMode::Insert,
        InputMode::Normal => ViMode::Normal,
    };

    render_input(
        frame,
        InputProps {
            area,
            buffer: &ctx.state.input,
            theme: ctx.theme,
            placeholder,
            show_cursor: ctx.state.focus == Focus::Input
                && !matches!(
                    ctx.state.overlay,
                    OverlayState::ProviderPicker(_)
                        | OverlayState::ModelPicker(_)
                        | OverlayState::ConversationPicker(_)
                        | OverlayState::SlashCommands(_)
                ),
            focused: ctx.state.focus == Focus::Input,
            navigation_mode: ctx.state.config.ui.navigation_mode,
            vi_mode,
        },
    );
}

fn render_too_small(frame: &mut Frame<'_>, area: Rect, theme: &Theme) {
    let message = format!(
        "Terminal too small (min {}x{}). Resize to continue.",
        MIN_WIDTH, MIN_HEIGHT
    );
    let paragraph = Paragraph::new(Text::from(message))
        .block(Block::default().borders(Borders::ALL).title("LLM"))
        .style(theme.error);
    frame.render_widget(paragraph, area);
}

fn overlay_size(width: u16) -> (u16, u16) {
    if width < COMPACT_WIDTH {
        (OVERLAY_COMPACT_WIDTH, OVERLAY_COMPACT_HEIGHT)
    } else if width < 100 {
        (OVERLAY_MEDIUM_WIDTH, OVERLAY_MEDIUM_HEIGHT)
    } else {
        (OVERLAY_WIDTH_PCT, OVERLAY_HEIGHT_PCT)
    }
}

fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    const FULL_PERCENT: u16 = 100;
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((FULL_PERCENT - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((FULL_PERCENT - percent_y) / 2),
        ])
        .split(area);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((FULL_PERCENT - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((FULL_PERCENT - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}
