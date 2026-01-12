use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "llm",
    about = "Interactive CLI interface for chatting with LLM providers",
    allow_hyphen_values = true
)]
pub struct CliArgs {
    #[arg(index = 1)]
    pub command: Option<String>,
    #[arg(index = 2)]
    pub provider_or_key: Option<String>,
    #[arg(index = 3)]
    pub prompt_or_value: Option<String>,
    #[arg(long, short = 'p')]
    pub provider: Option<String>,
    #[arg(long, short = 'm')]
    pub model: Option<String>,
    #[arg(long)]
    pub system: Option<String>,
    #[arg(long)]
    pub api_key: Option<String>,
    #[arg(long)]
    pub base_url: Option<String>,
    #[arg(long)]
    pub temperature: Option<f32>,
    #[arg(long)]
    pub max_tokens: Option<u32>,
    #[arg(long, short = 'c')]
    pub config: Option<PathBuf>,
    #[arg(long)]
    pub conversation: Option<String>,
    #[arg(long)]
    pub new: bool,
    #[arg(long)]
    pub prompt: Option<String>,
    #[arg(long)]
    pub list_providers: bool,
    #[arg(long)]
    pub list_models: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommandKind {
    Set,
    Get,
    Delete,
    Default,
    Chat,
}

impl CommandKind {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.to_lowercase().as_str() {
            "set" => Some(Self::Set),
            "get" => Some(Self::Get),
            "delete" => Some(Self::Delete),
            "default" => Some(Self::Default),
            "chat" => Some(Self::Chat),
            _ => None,
        }
    }
}

impl CliArgs {
    pub fn command_kind(&self) -> Option<CommandKind> {
        self.command.as_deref().and_then(CommandKind::parse)
    }

    pub fn has_non_interactive_prompt(&self) -> bool {
        self.prompt.is_some() || self.prompt_or_value.is_some()
    }
}
