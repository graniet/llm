use super::theme::Theme;
use crate::conversation::{ConversationMessage, MessageId};
use crate::runtime::{CollapsibleState, ScrollState, TOOL_COLLAPSE_LINES};
use measure::wrapped_height;
use ratatui::layout::Rect;
use ratatui::style::{Modifier, Style};
use ratatui::text::Text;
use ratatui::widgets::{Paragraph, Wrap};
use ratatui::Frame;
use std::collections::HashMap;
use text::render_message_text;
mod measure;
mod text;
#[derive(Default)]
pub struct MessageRenderer {
    cache: HashMap<MessageId, RenderCacheEntry>,
}

struct RenderCacheEntry {
    width: u16,
    version: u64,
    text: Text<'static>,
    height: u16,
    collapsed: bool,
}

impl MessageRenderer {
    fn render(
        &mut self,
        message: &ConversationMessage,
        width: u16,
        theme: &Theme,
        collapse: &CollapsibleState,
    ) -> &RenderCacheEntry {
        let entry = self.cache.entry(message.id);
        let collapsed = collapse_for(message, collapse);
        match entry {
            std::collections::hash_map::Entry::Occupied(mut occupied) => {
                if needs_update(occupied.get(), width, message.version, collapsed) {
                    let updated = build_entry(message, width, theme, collapse);
                    occupied.insert(updated);
                }
                occupied.into_mut()
            }
            std::collections::hash_map::Entry::Vacant(vacant) => {
                vacant.insert(build_entry(message, width, theme, collapse))
            }
        }
    }
}
pub struct MessageRenderProps<'a> {
    pub area: Rect,
    pub messages: &'a [ConversationMessage],
    pub theme: &'a Theme,
    pub scroll: ScrollState,
    pub selected: Option<MessageId>,
    pub collapse: &'a CollapsibleState,
}
pub fn render_messages(
    frame: &mut Frame<'_>,
    renderer: &mut MessageRenderer,
    props: MessageRenderProps<'_>,
) {
    let segments = collect_segments(renderer, &props);
    let total_height: u16 = segments.iter().map(|seg| seg.height).sum();
    let start_y = props
        .area
        .y
        .saturating_add(props.area.height.saturating_sub(total_height));
    let mut y = start_y;
    for segment in segments.into_iter().rev() {
        let area = Rect::new(props.area.x, y, props.area.width, segment.height);
        let mut paragraph = Paragraph::new(segment.text).wrap(Wrap { trim: false });
        paragraph = paragraph.scroll((segment.scroll, 0));
        if props.selected == Some(segment.id) {
            paragraph = paragraph.style(Style::default().add_modifier(Modifier::REVERSED));
        }
        frame.render_widget(paragraph, area);
        y = y.saturating_add(segment.height);
    }
}
struct Segment {
    id: MessageId,
    text: Text<'static>,
    height: u16,
    scroll: u16,
}
fn collect_segments(
    renderer: &mut MessageRenderer,
    props: &MessageRenderProps<'_>,
) -> Vec<Segment> {
    let total_height = total_message_height(
        renderer,
        props.messages,
        props.area.width,
        props.theme,
        props.collapse,
    );
    let mut state = SegmentState::new(props.area.height, props.scroll, total_height);
    let mut segments = Vec::new();
    for message in props.messages.iter().rev() {
        if state.is_full() {
            break;
        }
        if let Some(segment) = state.segment_for(
            message,
            renderer,
            props.theme,
            props.area.width,
            props.collapse,
        ) {
            segments.push(segment);
        }
    }
    segments
}
fn total_message_height(
    renderer: &mut MessageRenderer,
    messages: &[ConversationMessage],
    width: u16,
    theme: &Theme,
    collapse: &CollapsibleState,
) -> i32 {
    messages
        .iter()
        .map(|msg| renderer.render(msg, width, theme, collapse).height as i32)
        .sum::<i32>()
        .max(0)
}
struct SegmentState {
    remaining: i32,
    skip: i32,
}
impl SegmentState {
    fn new(area_height: u16, scroll: ScrollState, total_height: i32) -> Self {
        let remaining = area_height as i32;
        let max_offset = total_height.saturating_sub(remaining);
        let skip = (scroll.offset() as i32).min(max_offset);
        Self { remaining, skip }
    }
    fn is_full(&self) -> bool {
        self.remaining <= 0
    }
    fn segment_for(
        &mut self,
        message: &ConversationMessage,
        renderer: &mut MessageRenderer,
        theme: &Theme,
        width: u16,
        collapse: &CollapsibleState,
    ) -> Option<Segment> {
        let entry = renderer.render(message, width, theme, collapse);
        let height = entry.height as i32;
        if self.skip >= height {
            self.skip -= height;
            return None;
        }
        let available = height - self.skip;
        let take = available.min(self.remaining);
        let scroll_top = (available - take) as u16;
        self.remaining -= take;
        self.skip = 0;
        Some(Segment {
            id: message.id,
            text: entry.text.clone(),
            height: take as u16,
            scroll: scroll_top,
        })
    }
}
fn needs_update(entry: &RenderCacheEntry, width: u16, version: u64, collapsed: bool) -> bool {
    entry.width != width || entry.version != version || entry.collapsed != collapsed
}
fn build_entry(
    message: &ConversationMessage,
    width: u16,
    theme: &Theme,
    collapse: &CollapsibleState,
) -> RenderCacheEntry {
    let collapsed = collapse_for(message, collapse);
    let text = render_message_text(message, theme, collapsed);
    let height = wrapped_height(&text, width);
    RenderCacheEntry {
        width,
        version: message.version,
        text,
        height: height.max(1),
        collapsed,
    }
}
fn collapse_for(message: &ConversationMessage, collapse: &CollapsibleState) -> bool {
    let crate::conversation::MessageKind::ToolResult(result) = &message.kind else {
        return false;
    };
    if result.output.lines().count() <= TOOL_COLLAPSE_LINES {
        return false;
    }
    !collapse.is_expanded(message.id)
}
