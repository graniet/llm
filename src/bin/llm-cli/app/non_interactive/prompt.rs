use std::io::IsTerminal;
use std::io::{self, Read};

use crate::args::CliArgs;

use super::super::utils::parse_provider_and_model;

pub(super) fn resolve_prompt(args: &CliArgs) -> anyhow::Result<String> {
    if let Some(prompt) = prompt_from_flags(args) {
        return Ok(prompt);
    }
    if let Some(prompt) = prompt_from_positionals(args) {
        return Ok(prompt);
    }
    if let Some(prompt) = prompt_from_stdin()? {
        return Ok(prompt);
    }
    Err(anyhow::anyhow!(
        "no prompt provided; use --prompt or pipe input"
    ))
}

fn prompt_from_flags(args: &CliArgs) -> Option<String> {
    args.prompt.clone().or(args.prompt_or_value.clone())
}

fn prompt_from_positionals(args: &CliArgs) -> Option<String> {
    if args.command_kind().is_some() {
        return None;
    }
    let command = args.command.as_deref()?;
    if parse_provider_and_model(Some(command)).is_some() {
        return args.provider_or_key.clone();
    }
    Some(command.to_string())
}

fn prompt_from_stdin() -> anyhow::Result<Option<String>> {
    if io::stdin().is_terminal() {
        return Ok(None);
    }
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let trimmed = input.trim_end();
    if trimmed.is_empty() {
        Ok(None)
    } else {
        Ok(Some(trimmed.to_string()))
    }
}
