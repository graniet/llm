mod actions;
mod backtrack;
mod collapsible;
mod config;
mod context;
mod dialogue;
mod diff;
mod extras;
mod focus;
mod input;
mod notice;
mod onboarding;
mod skills;
mod slash;
mod status;
mod stream;
mod tool;
mod tools;

use tokio::sync::mpsc;
use tokio::sync::oneshot;

use crate::config::ConfigPaths;
use crate::runtime::StreamManager;
use crate::runtime::{AppEvent, AppState};
use crate::tools::{ToolContext, ToolRegistry};

pub struct AppController {
    pub state: AppState,
    pub stream_manager: StreamManager,
    pub event_sender: mpsc::Sender<AppEvent>,
    pub tool_registry: ToolRegistry,
    pub tool_context: ToolContext,
    pub config_paths: ConfigPaths,
    pub pending_tool_approval: Option<oneshot::Sender<bool>>,
    pub pending_tool_calls: std::collections::HashMap<crate::conversation::ConversationId, usize>,
}

pub struct AppControllerParams {
    pub state: AppState,
    pub stream_manager: StreamManager,
    pub event_sender: mpsc::Sender<AppEvent>,
    pub tool_registry: ToolRegistry,
    pub tool_context: ToolContext,
    pub config_paths: ConfigPaths,
}

impl AppController {
    pub fn new(params: AppControllerParams) -> Self {
        let AppControllerParams {
            state,
            stream_manager,
            event_sender,
            tool_registry,
            tool_context,
            config_paths,
        } = params;
        Self {
            state,
            stream_manager,
            event_sender,
            tool_registry,
            tool_context,
            config_paths,
            pending_tool_approval: None,
            pending_tool_calls: std::collections::HashMap::new(),
        }
    }

    pub async fn handle_event(&mut self, event: AppEvent) -> bool {
        match event {
            AppEvent::Input(input) => input::handle_input(self, input).await,
            AppEvent::Stream(ev) => stream::handle_stream(self, ev).await,
            AppEvent::Tool(ev) => tool::handle_tool(self, ev).await,
            AppEvent::Tick => input::handle_tick(self),
        }
    }
}
