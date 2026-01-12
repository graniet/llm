use std::io::{self, Write};

use llm::chat::ChatMessage;

use crate::config::ToolExecutionMode;
use crate::provider::ProviderHandle;
use crate::tools::{ToolContext, ToolRegistry};

use super::stream::{stream_once, StreamOutcome};
use super::tooling::ToolRunner;

pub(super) struct NonInteractiveRunner {
    handle: ProviderHandle,
    tool_runner: ToolRunner,
}

impl NonInteractiveRunner {
    pub(super) fn new(
        handle: ProviderHandle,
        tool_registry: ToolRegistry,
        tool_context: ToolContext,
        execution_mode: ToolExecutionMode,
    ) -> Self {
        Self {
            handle,
            tool_runner: ToolRunner::new(tool_registry, tool_context, execution_mode),
        }
    }

    pub(super) async fn run(&mut self, prompt: String) -> anyhow::Result<()> {
        let mut messages = vec![ChatMessage::user().content(prompt).build()];
        loop {
            let outcome = stream_once(&self.handle, &messages).await?;
            if self.apply_outcome(&mut messages, outcome)? {
                return Ok(());
            }
        }
    }

    fn apply_outcome(
        &mut self,
        messages: &mut Vec<ChatMessage>,
        outcome: StreamOutcome,
    ) -> anyhow::Result<bool> {
        if let Some(msg) = assistant_message(&outcome.text) {
            messages.push(msg);
        }
        if outcome.tool_calls.is_empty() {
            finish_output()?;
            return Ok(true);
        }
        self.tool_runner.print_calls(&outcome.tool_calls);
        messages.push(
            ChatMessage::assistant()
                .tool_use(outcome.tool_calls.clone())
                .build(),
        );
        let results = self.tool_runner.execute(&outcome.tool_calls)?;
        self.tool_runner.print_results(&results);
        messages.push(ChatMessage::assistant().tool_result(results).build());
        Ok(false)
    }
}

fn assistant_message(text: &str) -> Option<ChatMessage> {
    if text.trim().is_empty() {
        None
    } else {
        Some(ChatMessage::assistant().content(text).build())
    }
}

fn finish_output() -> anyhow::Result<()> {
    let mut stdout = io::stdout();
    stdout.write_all(b"\n")?;
    stdout.flush()?;
    Ok(())
}
