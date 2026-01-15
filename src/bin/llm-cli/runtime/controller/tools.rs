use crate::runtime::{OverlayState, PickerItem, PickerState, ToolBuilderState};

use super::AppController;

impl AppController {
    /// Open the tool builder wizard overlay
    pub fn open_tool_builder(&mut self) -> bool {
        self.state.overlay = OverlayState::ToolBuilder(ToolBuilderState::new());
        true
    }

    /// Open the tool picker overlay showing all available tools
    pub fn open_tool_picker(&mut self) -> bool {
        let items: Vec<PickerItem> = self
            .tool_registry
            .tool_names()
            .into_iter()
            .map(|name| {
                let is_user_tool = self.is_user_tool(name);
                let badges = if is_user_tool {
                    vec!["custom".to_string()]
                } else {
                    vec!["builtin".to_string()]
                };
                PickerItem {
                    id: name.to_string(),
                    label: name.to_string(),
                    meta: None,
                    badges,
                }
            })
            .collect();

        if items.is_empty() {
            self.push_notice("No tools available");
            return false;
        }

        self.state.overlay = OverlayState::ToolPicker(PickerState::new("Tools", items));
        true
    }

    /// Check if a tool is a user-defined tool
    fn is_user_tool(&self, name: &str) -> bool {
        let path = self.config_paths.user_tools_file();
        if let Ok(config) = crate::tools::UserToolsConfig::load(&path) {
            return config.get_tool(name).is_some();
        }
        false
    }

    /// Save a new tool from the builder wizard
    pub fn save_tool_from_builder(&mut self, draft: crate::runtime::UserToolDraft) -> bool {
        let tool: crate::tools::UserTool = draft.into();
        let tool_name = tool.name.clone();
        let path = self.config_paths.user_tools_file();

        let mut config = crate::tools::UserToolsConfig::load(&path).unwrap_or_default();
        config.add_tool(tool);

        if let Err(e) = config.save(&path) {
            self.set_status(crate::runtime::AppStatus::Error(format!(
                "Failed to save tool: {e}"
            )));
            return false;
        }

        // Reload tools in registry
        self.tool_registry.load_user_tools(&path);
        self.push_notice(&format!("Tool '{}' created successfully!", tool_name));
        true
    }

    /// Remove a user-defined tool
    pub fn remove_user_tool(&mut self, name: &str) -> bool {
        let path = self.config_paths.user_tools_file();

        let mut config = match crate::tools::UserToolsConfig::load(&path) {
            Ok(c) => c,
            Err(e) => {
                self.set_status(crate::runtime::AppStatus::Error(format!(
                    "Failed to load tools: {e}"
                )));
                return false;
            }
        };

        if config.remove_tool(name) {
            if let Err(e) = config.save(&path) {
                self.set_status(crate::runtime::AppStatus::Error(format!(
                    "Failed to save: {e}"
                )));
                return false;
            }
            // Reload registry (will remove the tool)
            self.tool_registry = crate::tools::ToolRegistry::from_config(&self.state.config.tools);
            self.tool_registry.load_user_tools(&path);
            self.push_notice(&format!("Tool '{}' removed", name));
            true
        } else {
            self.set_status(crate::runtime::AppStatus::Error(format!(
                "Tool '{}' not found or is builtin",
                name
            )));
            false
        }
    }
}
