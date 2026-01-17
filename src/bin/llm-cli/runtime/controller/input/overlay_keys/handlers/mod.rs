mod backtrack;
mod confirm;
mod dialogue_builder;
mod diff;
mod help;
mod onboarding;
mod pager;
mod picker;
mod scroll;
mod search;
mod tool_approval;
mod tool_builder;
mod tool_picker;

use crossterm::event::KeyEvent;

use crate::runtime::overlay::DialogueBuilderState;
use crate::runtime::{
    BacktrackOverlayState, DiffViewerState, OnboardingState, PagerState, PickerState, SearchState,
    ToolBuilderState,
};

use super::OverlayResult;

pub(super) fn handle_backtrack(state: &mut BacktrackOverlayState, key: KeyEvent) -> OverlayResult {
    backtrack::handle_backtrack(state, key)
}

pub(super) fn handle_confirm(key: KeyEvent) -> OverlayResult {
    confirm::handle_confirm(key)
}

pub(super) fn handle_diff_viewer(
    state: &mut DiffViewerState,
    key: KeyEvent,
    height: u16,
) -> OverlayResult {
    diff::handle_diff_viewer(state, key, height)
}

pub(super) fn handle_help(key: KeyEvent) -> OverlayResult {
    help::handle_help(key)
}

pub(super) fn handle_onboarding(state: &mut OnboardingState, key: KeyEvent) -> OverlayResult {
    onboarding::handle_onboarding(state, key)
}

pub(super) fn handle_pager(state: &mut PagerState, key: KeyEvent, height: u16) -> OverlayResult {
    pager::handle_pager(state, key, height)
}

pub(super) fn handle_picker(state: &mut PickerState, key: KeyEvent) -> OverlayResult {
    picker::handle_picker(state, key)
}

pub(super) fn handle_model_picker(state: &mut PickerState, key: KeyEvent) -> OverlayResult {
    picker::handle_model_picker(state, key)
}

pub(super) fn handle_skill_picker(state: &mut PickerState, key: KeyEvent) -> OverlayResult {
    picker::handle_skill_picker(state, key)
}

pub(super) fn handle_search(state: &mut SearchState, key: KeyEvent) -> OverlayResult {
    search::handle_search(state, key)
}

pub(super) fn handle_tool_approval(key: KeyEvent) -> OverlayResult {
    tool_approval::handle_tool_approval(key)
}

pub(super) fn handle_tool_builder(state: &mut ToolBuilderState, key: KeyEvent) -> OverlayResult {
    tool_builder::handle_tool_builder(state, key)
}

pub(super) fn handle_tool_picker(state: &mut PickerState, key: KeyEvent) -> OverlayResult {
    tool_picker::handle_tool_picker(state, key)
}

pub(super) fn handle_dialogue_builder(
    state: &mut DialogueBuilderState,
    key: KeyEvent,
) -> OverlayResult {
    dialogue_builder::handle_dialogue_builder(state, key)
}
