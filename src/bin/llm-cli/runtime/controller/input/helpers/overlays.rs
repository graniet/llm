use crate::conversation::ConversationId;
use crate::provider::ProviderCapabilities;
use crate::runtime::{
    ConfirmState, OverlayState, PickerItem, PickerState, SearchState, SlashCommandState,
};

use super::super::AppController;

pub fn confirm_exit(controller: &mut AppController) -> bool {
    controller.state.overlay = OverlayState::ConfirmExit(ConfirmState {
        message: "Quit the app?".to_string(),
    });
    true
}

pub fn open_provider_picker(controller: &mut AppController) -> bool {
    let items = controller
        .state
        .provider_registry
        .list()
        .map(|info| PickerItem {
            id: info.id.to_string(),
            label: info.display_name.clone(),
            meta: None,
            badges: provider_badges(info.capabilities),
        })
        .collect();
    controller.state.overlay = OverlayState::ProviderPicker(PickerState::new("Providers", items));
    true
}

pub fn open_model_picker(controller: &mut AppController) -> bool {
    let items = model_items(controller);
    controller.state.overlay = OverlayState::ModelPicker(PickerState::new("Models", items));
    true
}

pub fn open_conversation_picker(controller: &mut AppController) -> bool {
    let items = controller
        .state
        .conversations
        .list()
        .map(|conv| PickerItem {
            id: conv.id.to_string(),
            label: conv.title.clone(),
            meta: Some(conv.provider_id.to_string()),
            badges: Vec::new(),
        })
        .collect();
    controller.state.overlay =
        OverlayState::ConversationPicker(PickerState::new("Conversations", items));
    true
}

pub fn open_help(controller: &mut AppController) -> bool {
    controller.state.overlay = OverlayState::Help;
    true
}

pub fn open_search(controller: &mut AppController) -> bool {
    controller.state.overlay = OverlayState::Search(SearchState::new());
    true
}

pub fn open_slash_commands(controller: &mut AppController) -> bool {
    controller.state.overlay = OverlayState::SlashCommands(SlashCommandState::new());
    true
}

pub fn open_skill_picker(controller: &mut AppController) -> bool {
    let items = controller
        .state
        .skills
        .list()
        .iter()
        .map(|skill| PickerItem {
            id: skill.name.clone(),
            label: skill.name.clone(),
            meta: Some(skill.description.clone()),
            badges: skill.aliases.clone(),
        })
        .collect();
    controller.state.overlay = OverlayState::SkillPicker(PickerState::new("Skills", items));
    true
}

pub fn apply_picker_selection(controller: &mut AppController, item: &PickerItem) {
    if let Ok(uuid) = uuid::Uuid::parse_str(&item.id) {
        let id = ConversationId::from(uuid);
        controller.state.conversations.set_active(id);
        return;
    }
    controller.switch_provider(item.id.clone());
    open_model_picker(controller);
}

pub fn resolve_tool_approval(controller: &mut AppController, approved: bool) {
    if let Some(sender) = controller.pending_tool_approval.take() {
        let _ = sender.send(approved);
    }
    controller.state.overlay = OverlayState::None;
}

fn model_items(controller: &AppController) -> Vec<PickerItem> {
    let provider_id = controller
        .state
        .active_conversation()
        .map(|conv| conv.provider_id.to_string())
        .unwrap_or_default();
    let models = controller
        .state
        .config
        .providers
        .get(&provider_id)
        .map(|cfg| cfg.models.clone())
        .unwrap_or_default();
    models
        .into_iter()
        .map(|model| PickerItem {
            badges: model_badges(&model),
            id: model.id.clone(),
            label: model.id,
            meta: model.context_window.map(|v| format!("{v} ctx")),
        })
        .collect()
}

fn provider_badges(caps: ProviderCapabilities) -> Vec<String> {
    let mut badges = Vec::new();
    if caps.streaming {
        badges.push("stream".to_string());
    }
    if caps.tools {
        badges.push("tools".to_string());
    }
    if caps.tool_streaming {
        badges.push("tool-stream".to_string());
    }
    if caps.vision {
        badges.push("vision".to_string());
    }
    if caps.models_list {
        badges.push("models".to_string());
    }
    badges
}

fn model_badges(model: &crate::config::ModelConfig) -> Vec<String> {
    let mut badges = Vec::new();
    if model.supports_streaming.unwrap_or(false) {
        badges.push("stream".to_string());
    }
    if model.supports_tools.unwrap_or(false) {
        badges.push("tools".to_string());
    }
    if model.supports_vision.unwrap_or(false) {
        badges.push("vision".to_string());
    }
    badges
}
