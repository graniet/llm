mod launch;
mod options;

pub use options::TuiOptions;

use crate::config::{AppConfig, ConfigPaths};
use crate::provider::ProviderRegistry;
use crate::runtime::{controller::AppController, init_terminal, restore_terminal, run_app};
use crate::runtime::{AppState, StreamManager};
use crate::skills::SkillCatalog;
use crate::terminal::TerminalCapabilities;
use crate::tools::{ToolContext, ToolRegistry};

use launch::apply_launch_options;

pub async fn run_tui(
    config: AppConfig,
    paths: ConfigPaths,
    registry: ProviderRegistry,
    options: TuiOptions,
) -> anyhow::Result<()> {
    let terminal_caps = TerminalCapabilities::detect();
    let mut terminal = init_terminal(&terminal_caps)?;
    let ctx = TuiContext::new(config, paths, registry, options, terminal_caps);
    let result = run_inner(ctx, &mut terminal).await;
    restore_terminal()?;
    result
}

struct TuiContext {
    config: AppConfig,
    paths: ConfigPaths,
    registry: ProviderRegistry,
    options: TuiOptions,
    terminal_caps: TerminalCapabilities,
}

impl TuiContext {
    fn new(
        config: AppConfig,
        paths: ConfigPaths,
        registry: ProviderRegistry,
        options: TuiOptions,
        terminal_caps: TerminalCapabilities,
    ) -> Self {
        Self {
            config,
            paths,
            registry,
            options,
            terminal_caps,
        }
    }
}

async fn run_inner(
    ctx: TuiContext,
    terminal: &mut crate::runtime::AppTerminal,
) -> anyhow::Result<()> {
    let bundle = build_controller(ctx)?;
    let mut controller = bundle.controller;
    load_conversations(&mut controller)?;
    apply_launch_options(&mut controller, &bundle.options)?;
    run_app(controller, terminal, bundle.rx, bundle.tx).await
}

struct ControllerBundle {
    controller: AppController,
    rx: tokio::sync::mpsc::Receiver<crate::runtime::AppEvent>,
    tx: tokio::sync::mpsc::Sender<crate::runtime::AppEvent>,
    options: TuiOptions,
}

fn build_controller(ctx: TuiContext) -> anyhow::Result<ControllerBundle> {
    let store =
        crate::persistence::JsonConversationStore::new(ctx.paths.data_dir.join("conversations"));
    let mut tool_registry = ToolRegistry::from_config(&ctx.config.tools);
    // Load user-defined tools from config
    tool_registry.load_user_tools(&ctx.paths.user_tools_file());
    let tool_context = ToolContext {
        allowed_paths: ctx.config.tools.allowed_paths.clone(),
        timeout_ms: ctx.config.tools.timeout_ms,
        working_dir: std::env::current_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| ".".to_string()),
    };
    let mut state = AppState::new(ctx.config, ctx.registry, store, ctx.terminal_caps);
    let skills_dir = ctx.paths.config_dir.join("skills");
    state.skills = SkillCatalog::load(&skills_dir)
        .map_err(|err| anyhow::anyhow!("failed to load skills: {err}"))?;
    let mut options = ctx.options;
    state.session_overrides = std::mem::take(&mut options.session_overrides);
    let (tx, rx) = tokio::sync::mpsc::channel(128);
    let stream_manager = StreamManager::new(tx.clone());
    let params = crate::runtime::controller::AppControllerParams {
        state,
        stream_manager,
        event_sender: tx.clone(),
        tool_registry,
        tool_context,
        config_paths: ctx.paths.clone(),
    };
    Ok(ControllerBundle {
        controller: AppController::new(params),
        rx,
        tx,
        options,
    })
}

fn load_conversations(controller: &mut AppController) -> anyhow::Result<()> {
    let mut conversations = controller.state.store.load_all()?;
    conversations.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
    for conv in conversations {
        controller.state.conversations.add(conv);
    }
    Ok(())
}
