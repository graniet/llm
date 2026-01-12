use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::config::NavigationMode;
use crate::runtime::{Focus, OverlayState};

use super::focus_keys;
use super::helpers;
use super::overlay_keys;
use super::simple_keys;
use super::slash_overlay;
use super::vi_keys;
use super::AppController;

pub async fn dispatch_key(controller: &mut AppController, key: KeyEvent) -> bool {
    if matches!(controller.state.overlay, OverlayState::SlashCommands(_)) {
        return slash_overlay::handle_slash_overlay(controller, key).await;
    }
    if matches!(controller.state.overlay, OverlayState::None) {
        return handle_main_key(controller, key).await;
    }
    overlay_keys::handle_overlay_key(controller, key).await
}

async fn handle_main_key(controller: &mut AppController, key: KeyEvent) -> bool {
    if controller.state.status.is_busy() && key.code == KeyCode::Esc {
        controller.cancel_active_stream();
        return true;
    }
    if key.code == KeyCode::Tab {
        return controller.toggle_focus();
    }
    if controller.state.focus == Focus::Messages {
        return focus_keys::handle_focus_messages(controller, key).await;
    }
    if key.modifiers.contains(KeyModifiers::CONTROL) {
        return handle_ctrl_key(controller, key).await;
    }
    match controller.state.config.ui.navigation_mode {
        NavigationMode::Simple => simple_keys::handle_simple_key(controller, key).await,
        NavigationMode::Vi => vi_keys::handle_vi_key(controller, key).await,
    }
}

async fn handle_ctrl_key(controller: &mut AppController, key: KeyEvent) -> bool {
    match key.code {
        KeyCode::Char('c') => helpers::confirm_exit(controller),
        KeyCode::Char('n') => helpers::start_new_conversation(controller),
        KeyCode::Char('s') => helpers::save_active_conversation(controller),
        KeyCode::Char('f') => helpers::fork_conversation(controller),
        KeyCode::Char('p') => helpers::open_provider_picker(controller),
        KeyCode::Char('o') => helpers::open_conversation_picker(controller),
        KeyCode::Char('l') => helpers::clear_screen(controller),
        KeyCode::Char('t') => controller.toggle_all_tool_outputs(),
        KeyCode::Char('z') => controller.open_backtrack(),
        KeyCode::Char('u') => helpers::page_up(controller),
        KeyCode::Char('d') => helpers::page_down(controller),
        KeyCode::Enter => controller.send_user_message().await,
        _ => false,
    }
}
