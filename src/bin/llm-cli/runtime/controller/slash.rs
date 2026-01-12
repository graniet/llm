use crate::config::{save_config, NavigationMode};
use crate::runtime::{AppStatus, InputMode, SlashCommandId};

use super::input::helpers;
use super::AppController;

impl AppController {
    pub async fn handle_slash_command(&mut self, command: SlashCommandId) -> bool {
        self.handle_slash_command_with_arg(command, None).await
    }

    pub async fn handle_slash_input(&mut self, input: &str) -> Option<bool> {
        let parsed = parse_slash_input(input)?;
        Some(
            self.handle_slash_command_with_arg(parsed.command, parsed.arg)
                .await,
        )
    }

    async fn handle_slash_command_with_arg(
        &mut self,
        command: SlashCommandId,
        arg: Option<&str>,
    ) -> bool {
        match command {
            SlashCommandId::Model => handle_model(self, arg),
            SlashCommandId::Provider => handle_provider(self, arg),
            SlashCommandId::New => helpers::start_new_conversation(self),
            SlashCommandId::Save => helpers::save_active_conversation(self),
            SlashCommandId::Load | SlashCommandId::List | SlashCommandId::Resume => {
                helpers::open_conversation_picker(self)
            }
            SlashCommandId::Branches => self.open_branches(),
            SlashCommandId::Config => self.open_config_overlay(),
            SlashCommandId::Theme => handle_theme(self, arg),
            SlashCommandId::Mode => handle_mode(self, arg),
            SlashCommandId::Copy => self.copy_selected(),
            SlashCommandId::Edit => self.edit_last_user(),
            SlashCommandId::Regenerate => self.regenerate_last().await,
            SlashCommandId::Delete => self.delete_selected(),
            SlashCommandId::Undo => self.open_backtrack(),
            SlashCommandId::Status => self.show_context_status(),
            SlashCommandId::Summarize => self.summarize_context(parse_count(arg)),
            SlashCommandId::Compact => self.compact_context(),
            SlashCommandId::Skill => handle_skill(self, arg),
            SlashCommandId::Tool => handle_tool(self, arg),
            SlashCommandId::ToolAdd => self.open_tool_builder(),
            SlashCommandId::ToolList => self.open_tool_picker(),
            SlashCommandId::ToolRemove => handle_tool_remove(self, arg),
            SlashCommandId::Help | SlashCommandId::Keys => helpers::open_help(self),
        }
    }
}

struct ParsedSlash<'a> {
    command: SlashCommandId,
    arg: Option<&'a str>,
}

fn parse_slash_input(input: &str) -> Option<ParsedSlash<'_>> {
    let trimmed = input.trim();
    let rest = trimmed.strip_prefix('/')?;
    let (name, arg) = rest.split_once(' ').unwrap_or((rest, ""));
    let command = SlashCommandId::from_name(name)?;
    let arg = arg.trim();
    let arg = if arg.is_empty() { None } else { Some(arg) };
    Some(ParsedSlash { command, arg })
}

fn handle_model(controller: &mut AppController, arg: Option<&str>) -> bool {
    if let Some(model) = arg {
        controller.set_model(model.to_string());
        return true;
    }
    helpers::open_model_picker(controller)
}

fn handle_provider(controller: &mut AppController, arg: Option<&str>) -> bool {
    if let Some(provider) = arg {
        controller.switch_provider(provider.to_string());
        return helpers::open_model_picker(controller);
    }
    helpers::open_provider_picker(controller)
}

fn handle_mode(controller: &mut AppController, arg: Option<&str>) -> bool {
    let next = match arg.and_then(parse_navigation_mode) {
        Some(mode) => mode,
        None => toggle_mode(controller.state.config.ui.navigation_mode),
    };
    controller.state.config.ui.navigation_mode = next;
    if next == NavigationMode::Simple {
        controller.state.input_mode = InputMode::Insert;
    }
    if let Err(err) = save_config(&controller.state.config, &controller.config_paths) {
        controller.set_status(AppStatus::Error(format!("save config: {err}")));
    }
    true
}

fn handle_skill(controller: &mut AppController, arg: Option<&str>) -> bool {
    if let Some(name) = arg {
        if let Some(skill) = controller.find_skill(name).cloned() {
            return controller.activate_skill(&skill);
        }
    }
    helpers::open_skill_picker(controller)
}

fn handle_theme(controller: &mut AppController, arg: Option<&str>) -> bool {
    let next = match arg {
        Some(value) => {
            if !is_known_theme(value) {
                controller.set_status(AppStatus::Error("unknown theme".to_string()));
                return false;
            }
            value.to_string()
        }
        None => toggle_theme(&controller.state.config.ui.theme),
    };
    controller.state.config.ui.theme = next;
    if let Err(err) = save_config(&controller.state.config, &controller.config_paths) {
        controller.set_status(AppStatus::Error(format!("save config: {err}")));
    }
    true
}

fn parse_navigation_mode(input: &str) -> Option<NavigationMode> {
    match input.trim().to_lowercase().as_str() {
        "simple" => Some(NavigationMode::Simple),
        "vi" => Some(NavigationMode::Vi),
        _ => None,
    }
}

fn toggle_mode(current: NavigationMode) -> NavigationMode {
    match current {
        NavigationMode::Simple => NavigationMode::Vi,
        NavigationMode::Vi => NavigationMode::Simple,
    }
}

fn toggle_theme(current: &str) -> String {
    match current.to_lowercase().as_str() {
        "codex" => "mono".to_string(),
        _ => "codex".to_string(),
    }
}

fn is_known_theme(name: &str) -> bool {
    matches!(name.to_lowercase().as_str(), "codex" | "mono")
}

fn parse_count(arg: Option<&str>) -> Option<usize> {
    arg.and_then(|value| value.trim().parse::<usize>().ok())
}

fn handle_tool(controller: &mut AppController, arg: Option<&str>) -> bool {
    match arg {
        Some("add") => controller.open_tool_builder(),
        Some("list") | Some("ls") => controller.open_tool_picker(),
        Some(name) if name.starts_with("remove ") => {
            let tool_name = name.strip_prefix("remove ").unwrap_or("");
            handle_tool_remove(controller, Some(tool_name))
        }
        _ => controller.open_tool_picker(),
    }
}

fn handle_tool_remove(controller: &mut AppController, arg: Option<&str>) -> bool {
    let Some(name) = arg else {
        controller.set_status(AppStatus::Error("Usage: /tool-remove <name>".to_string()));
        return false;
    };

    let path = controller.config_paths.user_tools_file();
    match crate::tools::UserToolsConfig::load(&path) {
        Ok(mut config) => {
            if config.remove_tool(name) {
                if let Err(e) = config.save(&path) {
                    controller.set_status(AppStatus::Error(format!("Failed to save: {e}")));
                    return false;
                }
                // Reload tools in registry
                controller.tool_registry.load_user_tools(&path);
                controller.set_status(AppStatus::Idle);
                controller.push_notice(&format!("Tool '{}' removed", name));
                true
            } else {
                controller.set_status(AppStatus::Error(format!("Tool '{}' not found", name)));
                false
            }
        }
        Err(e) => {
            controller.set_status(AppStatus::Error(format!("Failed to load tools: {e}")));
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_slash_with_arg() {
        let parsed = parse_slash_input("/summarize 3").unwrap();
        assert_eq!(parsed.command, SlashCommandId::Summarize);
        assert_eq!(parsed.arg, Some("3"));
    }

    #[test]
    fn parses_branches_command() {
        let parsed = parse_slash_input("/branches").unwrap();
        assert_eq!(parsed.command, SlashCommandId::Branches);
        assert!(parsed.arg.is_none());
    }
}
