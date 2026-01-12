#[path = "llm-cli/app/mod.rs"]
mod app;
#[path = "llm-cli/args.rs"]
mod args;
#[path = "llm-cli/config/mod.rs"]
mod config;
#[path = "llm-cli/conversation/mod.rs"]
mod conversation;
#[path = "llm-cli/diff/mod.rs"]
mod diff;
#[path = "llm-cli/history/mod.rs"]
mod history;
#[path = "llm-cli/input/mod.rs"]
mod input;
#[path = "llm-cli/logging.rs"]
mod logging;
#[path = "llm-cli/model/mod.rs"]
mod model;
#[path = "llm-cli/persistence/mod.rs"]
mod persistence;
#[path = "llm-cli/provider/mod.rs"]
mod provider;
#[path = "llm-cli/runtime/mod.rs"]
mod runtime;
#[path = "llm-cli/skills/mod.rs"]
mod skills;
#[path = "llm-cli/terminal/mod.rs"]
mod terminal;
#[path = "llm-cli/tools/mod.rs"]
mod tools;
#[path = "llm-cli/ui/mod.rs"]
mod ui;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    app::run().await
}
