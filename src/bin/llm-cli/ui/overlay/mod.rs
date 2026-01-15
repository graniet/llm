mod backtrack;
mod confirm;
mod diff_viewer;
mod help;
mod onboarding;
mod pager;
mod picker;
mod search;
mod tool_approval;
mod tool_builder;

use ratatui::layout::Rect;
use ratatui::Frame;

use crate::runtime::OverlayState;

use super::slash_commands;
use super::theme::Theme;

pub fn render_overlay(frame: &mut Frame<'_>, area: Rect, overlay: &OverlayState, theme: &Theme) {
    match overlay {
        OverlayState::None => {}
        OverlayState::Help => help::render_help(frame, area, theme),
        OverlayState::ProviderPicker(state)
        | OverlayState::ModelPicker(state)
        | OverlayState::ConversationPicker(state)
        | OverlayState::SkillPicker(state)
        | OverlayState::ToolPicker(state) => picker::render_picker(frame, area, state, theme),
        OverlayState::Onboarding(state) => onboarding::render_onboarding(frame, area, state, theme),
        OverlayState::Pager(state) => pager::render_pager(frame, area, state, theme),
        OverlayState::Backtrack(state) => backtrack::render_backtrack(frame, area, state, theme),
        OverlayState::DiffViewer(state) => {
            diff_viewer::render_diff_viewer(frame, area, state, theme)
        }
        OverlayState::SlashCommands(state) => {
            slash_commands::render_slash_popup(frame, area, state, theme)
        }
        OverlayState::ConfirmExit(state) => confirm::render_confirm(frame, area, state, theme),
        OverlayState::ToolApproval(state) => {
            tool_approval::render_tool_approval(frame, area, state, theme)
        }
        OverlayState::ToolBuilder(state) => {
            tool_builder::render_tool_builder(frame, area, state, theme)
        }
        OverlayState::Search(state) => search::render_search(frame, area, state, theme),
    }
}
