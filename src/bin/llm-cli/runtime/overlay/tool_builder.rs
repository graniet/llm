/// Interactive tool builder wizard state
#[derive(Debug, Clone)]
pub struct ToolBuilderState {
    pub step: ToolBuilderStep,
    pub name: String,
    pub description: String,
    pub params: Vec<ToolParamDraft>,
    pub command: String,
    pub current_input: String,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolBuilderStep {
    Name,
    Description,
    ParamName,
    ParamType,
    ParamDesc,
    ParamRequired,
    AddMoreParams,
    Command,
    Confirm,
}

#[derive(Debug, Clone)]
pub struct ToolParamDraft {
    pub name: String,
    pub param_type: String,
    pub description: String,
    pub required: bool,
}

#[allow(dead_code)]
impl ToolBuilderState {
    pub fn new() -> Self {
        Self {
            step: ToolBuilderStep::Name,
            name: String::new(),
            description: String::new(),
            params: Vec::new(),
            command: String::new(),
            current_input: String::new(),
            error: None,
        }
    }

    pub fn step_title(&self) -> &'static str {
        match self.step {
            ToolBuilderStep::Name => "Tool Name",
            ToolBuilderStep::Description => "Description",
            ToolBuilderStep::ParamName => "Parameter Name",
            ToolBuilderStep::ParamType => "Parameter Type",
            ToolBuilderStep::ParamDesc => "Parameter Description",
            ToolBuilderStep::ParamRequired => "Required?",
            ToolBuilderStep::AddMoreParams => "Add More Parameters?",
            ToolBuilderStep::Command => "Shell Command",
            ToolBuilderStep::Confirm => "Confirm",
        }
    }

    pub fn step_prompt(&self) -> &'static str {
        match self.step {
            ToolBuilderStep::Name => "Enter tool name (e.g., 'git_status'):",
            ToolBuilderStep::Description => "Enter tool description:",
            ToolBuilderStep::ParamName => "Enter parameter name (or press Enter to skip):",
            ToolBuilderStep::ParamType => "Enter type (string/number/boolean):",
            ToolBuilderStep::ParamDesc => "Enter parameter description:",
            ToolBuilderStep::ParamRequired => "Is this parameter required? (y/n):",
            ToolBuilderStep::AddMoreParams => "Add another parameter? (y/n):",
            ToolBuilderStep::Command => "Enter shell command (use {{param}} for params):",
            ToolBuilderStep::Confirm => "Create this tool? (y/n):",
        }
    }

    pub fn push_char(&mut self, ch: char) {
        self.current_input.push(ch);
        self.error = None;
    }

    pub fn pop_char(&mut self) {
        self.current_input.pop();
        self.error = None;
    }

    /// Returns true if the wizard should close (completed or cancelled)
    pub fn submit(&mut self) -> ToolBuilderResult {
        let input = self.current_input.trim().to_string();

        match self.step {
            ToolBuilderStep::Name => {
                if input.is_empty() {
                    self.error = Some("Name cannot be empty".to_string());
                    return ToolBuilderResult::Continue;
                }
                if !is_valid_identifier(&input) {
                    self.error = Some("Name must be alphanumeric with underscores".to_string());
                    return ToolBuilderResult::Continue;
                }
                self.name = input;
                self.current_input.clear();
                self.step = ToolBuilderStep::Description;
            }
            ToolBuilderStep::Description => {
                if input.is_empty() {
                    self.error = Some("Description cannot be empty".to_string());
                    return ToolBuilderResult::Continue;
                }
                self.description = input;
                self.current_input.clear();
                self.step = ToolBuilderStep::ParamName;
            }
            ToolBuilderStep::ParamName => {
                if input.is_empty() {
                    // Skip to command step
                    self.step = ToolBuilderStep::Command;
                } else {
                    if !is_valid_identifier(&input) {
                        self.error = Some("Parameter name must be alphanumeric".to_string());
                        return ToolBuilderResult::Continue;
                    }
                    self.params.push(ToolParamDraft {
                        name: input,
                        param_type: String::new(),
                        description: String::new(),
                        required: false,
                    });
                    self.current_input.clear();
                    self.step = ToolBuilderStep::ParamType;
                }
            }
            ToolBuilderStep::ParamType => {
                let ptype = if input.is_empty() {
                    "string".to_string()
                } else {
                    match input.to_lowercase().as_str() {
                        "string" | "str" | "s" => "string".to_string(),
                        "number" | "num" | "n" | "int" | "integer" => "number".to_string(),
                        "boolean" | "bool" | "b" => "boolean".to_string(),
                        _ => {
                            self.error = Some("Type must be string, number, or boolean".to_string());
                            return ToolBuilderResult::Continue;
                        }
                    }
                };
                if let Some(param) = self.params.last_mut() {
                    param.param_type = ptype;
                }
                self.current_input.clear();
                self.step = ToolBuilderStep::ParamDesc;
            }
            ToolBuilderStep::ParamDesc => {
                if let Some(param) = self.params.last_mut() {
                    param.description = if input.is_empty() {
                        format!("The {} parameter", param.name)
                    } else {
                        input
                    };
                }
                self.current_input.clear();
                self.step = ToolBuilderStep::ParamRequired;
            }
            ToolBuilderStep::ParamRequired => {
                let required = matches!(input.to_lowercase().as_str(), "y" | "yes" | "true" | "1");
                if let Some(param) = self.params.last_mut() {
                    param.required = required;
                }
                self.current_input.clear();
                self.step = ToolBuilderStep::AddMoreParams;
            }
            ToolBuilderStep::AddMoreParams => {
                let add_more = matches!(input.to_lowercase().as_str(), "y" | "yes" | "true" | "1");
                self.current_input.clear();
                if add_more {
                    self.step = ToolBuilderStep::ParamName;
                } else {
                    self.step = ToolBuilderStep::Command;
                }
            }
            ToolBuilderStep::Command => {
                if input.is_empty() {
                    self.error = Some("Command cannot be empty".to_string());
                    return ToolBuilderResult::Continue;
                }
                self.command = input;
                self.current_input.clear();
                self.step = ToolBuilderStep::Confirm;
            }
            ToolBuilderStep::Confirm => {
                let confirm = matches!(input.to_lowercase().as_str(), "y" | "yes" | "true" | "1");
                if confirm {
                    return ToolBuilderResult::Complete(self.build_tool());
                } else {
                    return ToolBuilderResult::Cancel;
                }
            }
        }

        ToolBuilderResult::Continue
    }

    pub fn cancel(&self) -> ToolBuilderResult {
        ToolBuilderResult::Cancel
    }

    fn build_tool(&self) -> UserToolDraft {
        UserToolDraft {
            name: self.name.clone(),
            description: self.description.clone(),
            params: self.params.clone(),
            command: self.command.clone(),
        }
    }

    pub fn summary_lines(&self) -> Vec<String> {
        let mut lines = Vec::new();
        lines.push(format!("Name: {}", self.name));
        if !self.description.is_empty() {
            lines.push(format!("Description: {}", self.description));
        }
        if !self.params.is_empty() {
            lines.push("Parameters:".to_string());
            for param in &self.params {
                let req = if param.required { " (required)" } else { "" };
                lines.push(format!(
                    "  - {}: {} - {}{}",
                    param.name, param.param_type, param.description, req
                ));
            }
        }
        if !self.command.is_empty() {
            lines.push(format!("Command: {}", self.command));
        }
        lines
    }
}

#[derive(Debug, Clone)]
pub struct UserToolDraft {
    pub name: String,
    pub description: String,
    pub params: Vec<ToolParamDraft>,
    pub command: String,
}

#[derive(Debug)]
pub enum ToolBuilderResult {
    Continue,
    Complete(UserToolDraft),
    Cancel,
}

fn is_valid_identifier(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        && s.chars().next().map_or(false, |c| c.is_ascii_alphabetic())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_identifier() {
        assert!(is_valid_identifier("my_tool"));
        assert!(is_valid_identifier("getThing"));
        assert!(!is_valid_identifier("123tool"));
        assert!(!is_valid_identifier(""));
        assert!(!is_valid_identifier("has space"));
    }

    #[test]
    fn wizard_flow() {
        let mut state = ToolBuilderState::new();
        assert_eq!(state.step, ToolBuilderStep::Name);

        state.current_input = "my_tool".to_string();
        state.submit();
        assert_eq!(state.step, ToolBuilderStep::Description);

        state.current_input = "A test tool".to_string();
        state.submit();
        assert_eq!(state.step, ToolBuilderStep::ParamName);

        // Skip params
        state.current_input.clear();
        state.submit();
        assert_eq!(state.step, ToolBuilderStep::Command);

        state.current_input = "echo hello".to_string();
        state.submit();
        assert_eq!(state.step, ToolBuilderStep::Confirm);
    }
}
