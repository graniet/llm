use std::io::IsTerminal;
use std::io::{self, Write};

use llm::{FunctionCall, ToolCall};

use crate::config::ToolExecutionMode;
use crate::tools::{ToolContext, ToolRegistry};

pub(super) struct ToolRunner {
    registry: ToolRegistry,
    context: ToolContext,
    execution_mode: ToolExecutionMode,
    stdin_is_tty: bool,
}

impl ToolRunner {
    pub(super) fn new(
        registry: ToolRegistry,
        context: ToolContext,
        execution_mode: ToolExecutionMode,
    ) -> Self {
        Self {
            registry,
            context,
            execution_mode,
            stdin_is_tty: io::stdin().is_terminal(),
        }
    }

    pub(super) fn execute(&self, calls: &[ToolCall]) -> anyhow::Result<Vec<ToolCall>> {
        let mut results = Vec::with_capacity(calls.len());
        for call in calls {
            results.push(self.execute_call(call)?);
        }
        Ok(results)
    }

    pub(super) fn print_calls(&self, calls: &[ToolCall]) {
        for call in calls {
            eprintln!("[tool] {} {}", call.function.name, call.id);
            eprintln!("{}", call.function.arguments);
        }
    }

    pub(super) fn print_results(&self, results: &[ToolCall]) {
        for result in results {
            eprintln!("[tool result] {} {}", result.function.name, result.id);
            eprintln!("{}", result.function.arguments);
        }
    }

    fn execute_call(&self, call: &ToolCall) -> anyhow::Result<ToolCall> {
        let decision = self.should_execute_tool(call)?;
        let (output, success) = match decision {
            ToolDecision::Execute => self.run_tool(call),
            ToolDecision::Decline(reason) => (reason, false),
        };
        Ok(tool_result_call(call, output, success))
    }

    fn should_execute_tool(&self, call: &ToolCall) -> anyhow::Result<ToolDecision> {
        match self.execution_mode {
            ToolExecutionMode::Always => Ok(ToolDecision::Execute),
            ToolExecutionMode::Never => Ok(ToolDecision::Decline("Tool execution disabled".into())),
            ToolExecutionMode::Ask => self.ask_for_approval(call),
        }
    }

    fn ask_for_approval(&self, call: &ToolCall) -> anyhow::Result<ToolDecision> {
        if !self.stdin_is_tty {
            return Ok(ToolDecision::Decline(
                "Tool execution requires approval".into(),
            ));
        }
        let approved = prompt_for_tool(call)?;
        if approved {
            Ok(ToolDecision::Execute)
        } else {
            Ok(ToolDecision::Decline("Tool execution declined".into()))
        }
    }

    fn run_tool(&self, call: &ToolCall) -> (String, bool) {
        match self
            .registry
            .execute(&call.function.name, &call.function.arguments, &self.context)
        {
            Ok(output) => (output, true),
            Err(err) => (format!("Tool error: {err}"), false),
        }
    }
}

enum ToolDecision {
    Execute,
    Decline(String),
}

fn tool_result_call(call: &ToolCall, output: String, success: bool) -> ToolCall {
    let call_type = if success {
        call.call_type.clone()
    } else {
        "function".to_string()
    };
    ToolCall {
        id: call.id.clone(),
        call_type,
        function: FunctionCall {
            name: call.function.name.clone(),
            arguments: output,
        },
    }
}

fn prompt_for_tool(call: &ToolCall) -> anyhow::Result<bool> {
    eprintln!("[tool approval] {} {}", call.function.name, call.id);
    eprintln!("{}", call.function.arguments);
    eprint!("Run tool? [y/N]: ");
    io::stderr().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(matches!(input.trim(), "y" | "Y"))
}
