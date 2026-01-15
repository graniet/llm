mod handlers;

use crossterm::event::KeyEvent;

use crate::runtime::OverlayState;

use super::helpers;
use super::AppController;

use handlers::{
    handle_backtrack, handle_confirm, handle_diff_viewer, handle_help, handle_model_picker,
    handle_onboarding, handle_pager, handle_picker, handle_search, handle_skill_picker,
    handle_tool_approval, handle_tool_builder, handle_tool_picker,
};

pub async fn handle_overlay_key(controller: &mut AppController, key: KeyEvent) -> bool {
    let result = {
        let overlay = &mut controller.state.overlay;
        match overlay {
            OverlayState::Help => handle_help(key),
            OverlayState::ProviderPicker(state) => handle_picker(state, key),
            OverlayState::ModelPicker(state) => handle_model_picker(state, key),
            OverlayState::ConversationPicker(state) => handle_picker(state, key),
            OverlayState::SkillPicker(state) => handle_skill_picker(state, key),
            OverlayState::ToolPicker(state) => handle_tool_picker(state, key),
            OverlayState::Onboarding(state) => handle_onboarding(state, key),
            OverlayState::Pager(state) => {
                handle_pager(state, key, controller.state.terminal_size.1)
            }
            OverlayState::Backtrack(state) => handle_backtrack(state, key),
            OverlayState::DiffViewer(state) => {
                handle_diff_viewer(state, key, controller.state.terminal_size.1)
            }
            OverlayState::SlashCommands(_) => OverlayResult::action(OverlayAction::None),
            OverlayState::ConfirmExit(_) => handle_confirm(key),
            OverlayState::ToolApproval(_) => handle_tool_approval(key),
            OverlayState::ToolBuilder(state) => handle_tool_builder(state, key),
            OverlayState::Search(state) => handle_search(state, key),
            OverlayState::None => OverlayResult::action(OverlayAction::None),
        }
    };
    if result.close {
        controller.state.overlay = OverlayState::None;
    }
    apply_action(controller, result.action)
}

fn apply_action(controller: &mut AppController, action: OverlayAction) -> bool {
    match action {
        OverlayAction::None => false,
        OverlayAction::Handled => true,
        OverlayAction::Quit => {
            controller.state.should_quit = true;
            true
        }
        OverlayAction::FinishOnboarding => controller.finish_onboarding_from_overlay(),
        OverlayAction::PickerSelected(item) => {
            helpers::apply_picker_selection(controller, &item);
            true
        }
        OverlayAction::SetModel(model) => {
            controller.set_model(model);
            true
        }
        OverlayAction::ActivateSkill(name) => {
            if let Some(skill) = controller.find_skill(&name).cloned() {
                controller.activate_skill(&skill);
                return true;
            }
            false
        }
        OverlayAction::RestoreSnapshot(id) => controller.restore_snapshot(id),
        OverlayAction::ApplyDiff(diff) => controller.apply_diff(diff),
        OverlayAction::UpdateSearch => {
            controller.update_search_matches();
            true
        }
        OverlayAction::JumpToSearch => {
            controller.jump_to_search_match();
            true
        }
        OverlayAction::ToolApproval(approved) => {
            helpers::resolve_tool_approval(controller, approved);
            true
        }
        OverlayAction::ToolBuilderComplete(draft) => {
            controller.save_tool_from_builder(draft)
        }
        OverlayAction::ToolSelected(name) => {
            controller.push_notice(&format!("Tool: {} (use /tool-remove {} to delete)", name, name));
            true
        }
        OverlayAction::ToolRemove(name) => {
            controller.remove_user_tool(&name)
        }
    }
}

pub(super) struct OverlayResult {
    action: OverlayAction,
    close: bool,
}

impl OverlayResult {
    pub(super) fn action(action: OverlayAction) -> Self {
        Self {
            action,
            close: false,
        }
    }

    pub(super) fn close(action: OverlayAction) -> Self {
        Self {
            action,
            close: true,
        }
    }
}

pub(super) enum OverlayAction {
    None,
    Handled,
    Quit,
    FinishOnboarding,
    PickerSelected(crate::runtime::PickerItem),
    SetModel(String),
    ActivateSkill(String),
    RestoreSnapshot(crate::runtime::SnapshotId),
    ApplyDiff(crate::diff::DiffView),
    UpdateSearch,
    JumpToSearch,
    ToolApproval(bool),
    ToolBuilderComplete(crate::runtime::UserToolDraft),
    ToolSelected(String),
    ToolRemove(String),
}
