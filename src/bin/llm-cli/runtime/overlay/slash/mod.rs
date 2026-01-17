mod commands;

use fuzzy_matcher::skim::SkimMatcherV2;
use fuzzy_matcher::FuzzyMatcher;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SlashCategory {
    ModelProvider,
    Sessions,
    Configuration,
    MessageActions,
    Context,
    Skills,
    Tools,
    Dialogue,
    Help,
}

impl SlashCategory {
    pub fn label(self) -> &'static str {
        match self {
            SlashCategory::ModelProvider => "model/provider",
            SlashCategory::Sessions => "sessions",
            SlashCategory::Configuration => "configuration",
            SlashCategory::MessageActions => "messages",
            SlashCategory::Context => "context",
            SlashCategory::Skills => "skills",
            SlashCategory::Tools => "tools",
            SlashCategory::Dialogue => "dialogue",
            SlashCategory::Help => "help",
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SlashCommandId {
    Model,
    Provider,
    New,
    Save,
    Load,
    List,
    Resume,
    Branches,
    Config,
    Theme,
    Mode,
    Copy,
    Edit,
    Regenerate,
    Delete,
    Undo,
    Status,
    Summarize,
    Compact,
    Skill,
    Tool,
    ToolAdd,
    ToolList,
    ToolRemove,
    // Dialogue commands
    Multi,
    Invite,
    Kick,
    Stop,
    Continue,
    Help,
    Keys,
}

impl SlashCommandId {
    pub fn from_name(name: &str) -> Option<Self> {
        match name.trim().to_lowercase().as_str() {
            "model" => Some(Self::Model),
            "provider" => Some(Self::Provider),
            "new" => Some(Self::New),
            "save" => Some(Self::Save),
            "load" => Some(Self::Load),
            "list" => Some(Self::List),
            "resume" => Some(Self::Resume),
            "branches" => Some(Self::Branches),
            "config" => Some(Self::Config),
            "theme" => Some(Self::Theme),
            "mode" => Some(Self::Mode),
            "copy" => Some(Self::Copy),
            "edit" => Some(Self::Edit),
            "regenerate" => Some(Self::Regenerate),
            "delete" => Some(Self::Delete),
            "undo" => Some(Self::Undo),
            "status" => Some(Self::Status),
            "summarize" => Some(Self::Summarize),
            "compact" => Some(Self::Compact),
            "skill" => Some(Self::Skill),
            "tool" => Some(Self::Tool),
            "tool-add" | "tooladd" => Some(Self::ToolAdd),
            "tool-list" | "toollist" | "tools" => Some(Self::ToolList),
            "tool-remove" | "toolremove" => Some(Self::ToolRemove),
            "multi" | "dialogue" => Some(Self::Multi),
            "invite" => Some(Self::Invite),
            "kick" => Some(Self::Kick),
            "stop" => Some(Self::Stop),
            "continue" | "next" => Some(Self::Continue),
            "help" => Some(Self::Help),
            "keys" => Some(Self::Keys),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SlashCommand {
    pub id: SlashCommandId,
    pub name: &'static str,
    pub description: &'static str,
    pub category: SlashCategory,
    pub shortcut: Option<&'static str>,
    /// If set, selecting this command inserts the command into input instead of executing.
    /// The hint describes the expected argument (e.g., "<tool_name>").
    pub arg_hint: Option<&'static str>,
}

impl SlashCommand {
    pub const fn new(
        id: SlashCommandId,
        name: &'static str,
        description: &'static str,
        category: SlashCategory,
        shortcut: Option<&'static str>,
    ) -> Self {
        Self {
            id,
            name,
            description,
            category,
            shortcut,
            arg_hint: None,
        }
    }

    pub const fn with_arg(
        id: SlashCommandId,
        name: &'static str,
        description: &'static str,
        category: SlashCategory,
        arg_hint: &'static str,
    ) -> Self {
        Self {
            id,
            name,
            description,
            category,
            shortcut: None,
            arg_hint: Some(arg_hint),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SlashCommandState {
    pub query: String,
    pub commands: Vec<SlashCommand>,
    pub filtered: Vec<SlashCommand>,
    pub selected: usize,
}

impl SlashCommandState {
    pub fn new() -> Self {
        let commands = commands::default_commands();
        let mut state = Self {
            query: String::new(),
            filtered: commands.clone(),
            commands,
            selected: 0,
        };
        state.refresh();
        state
    }

    pub fn push_query(&mut self, ch: char) {
        self.query.push(ch);
        self.refresh();
    }

    pub fn pop_query(&mut self) {
        self.query.pop();
        self.refresh();
    }

    pub fn next(&mut self) {
        if !self.filtered.is_empty() {
            self.selected = (self.selected + 1).min(self.filtered.len() - 1);
        }
    }

    pub fn prev(&mut self) {
        if self.selected > 0 {
            self.selected = self.selected.saturating_sub(1);
        }
    }

    pub fn selected_command(&self) -> Option<SlashCommand> {
        self.filtered.get(self.selected).copied()
    }

    fn refresh(&mut self) {
        if self.query.is_empty() {
            self.filtered = self.commands.clone();
            return;
        }
        let matcher = SkimMatcherV2::default();
        let mut scored: Vec<(i64, SlashCommand)> = self
            .commands
            .iter()
            .filter_map(|cmd| {
                let score = matcher
                    .fuzzy_match(cmd.name, &self.query)
                    .or_else(|| matcher.fuzzy_match(cmd.description, &self.query))?;
                Some((score, *cmd))
            })
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0));
        self.filtered = scored.into_iter().map(|(_, cmd)| cmd).collect();
        self.selected = self.selected.min(self.filtered.len().saturating_sub(1));
    }
}
