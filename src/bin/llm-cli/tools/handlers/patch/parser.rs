//! Freeform patch parser using pest.

use pest::Parser;
use pest_derive::Parser;

use super::types::{Patch, PatchHunk};
use crate::tools::error::ToolError;

#[derive(Parser)]
#[grammar = "bin/llm-cli/tools/handlers/patch/patch.pest"]
struct PatchParser;

/// Parse a freeform Codex-style patch.
pub fn parse_freeform_patch(input: &str) -> Result<Patch, ToolError> {
    // Check for freeform marker
    if !input.contains("*** Begin Patch") {
        return Err(ToolError::InvalidArgs(
            "Freeform patch must start with '*** Begin Patch'".to_string(),
        ));
    }

    let pairs = PatchParser::parse(Rule::patch, input)
        .map_err(|e| ToolError::InvalidArgs(format!("Failed to parse patch: {e}")))?;

    let mut hunks = Vec::new();

    for pair in pairs {
        if pair.as_rule() == Rule::patch {
            for inner in pair.into_inner() {
                if inner.as_rule() == Rule::hunk {
                    if let Some(hunk) = parse_hunk(inner)? {
                        hunks.push(hunk);
                    }
                }
            }
        }
    }

    Ok(Patch { hunks })
}

fn parse_hunk(pair: pest::iterators::Pair<Rule>) -> Result<Option<PatchHunk>, ToolError> {
    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::add_hunk => return parse_add_hunk(inner).map(Some),
            Rule::delete_hunk => return parse_delete_hunk(inner).map(Some),
            Rule::update_hunk => return parse_update_hunk(inner).map(Some),
            _ => {}
        }
    }
    Ok(None)
}

fn parse_add_hunk(pair: pest::iterators::Pair<Rule>) -> Result<PatchHunk, ToolError> {
    let mut path = String::new();
    let mut lines = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::filename => {
                path = inner.as_str().to_string();
            }
            Rule::add_line => {
                for line_inner in inner.into_inner() {
                    if line_inner.as_rule() == Rule::line_content {
                        lines.push(line_inner.as_str().to_string());
                    }
                }
            }
            _ => {}
        }
    }

    Ok(PatchHunk::Add {
        path,
        content: lines.join("\n"),
    })
}

fn parse_delete_hunk(pair: pest::iterators::Pair<Rule>) -> Result<PatchHunk, ToolError> {
    let mut path = String::new();

    for inner in pair.into_inner() {
        if inner.as_rule() == Rule::filename {
            path = inner.as_str().to_string();
        }
    }

    Ok(PatchHunk::Delete { path })
}

fn parse_update_hunk(pair: pest::iterators::Pair<Rule>) -> Result<PatchHunk, ToolError> {
    let mut path = String::new();
    let mut new_path = None;
    let mut context_lines = Vec::new();
    let mut remove_lines = Vec::new();
    let mut add_lines = Vec::new();

    for inner in pair.into_inner() {
        match inner.as_rule() {
            Rule::filename => {
                path = inner.as_str().to_string();
            }
            Rule::change_move => {
                for move_inner in inner.into_inner() {
                    if move_inner.as_rule() == Rule::filename {
                        new_path = Some(move_inner.as_str().to_string());
                    }
                }
            }
            Rule::change => {
                for change_inner in inner.into_inner() {
                    match change_inner.as_rule() {
                        Rule::change_context => {
                            for ctx_inner in change_inner.into_inner() {
                                if ctx_inner.as_rule() == Rule::line_content {
                                    context_lines.push(ctx_inner.as_str().to_string());
                                }
                            }
                        }
                        Rule::change_line => {
                            let text = change_inner.as_str();
                            if let Some(first_char) = text.chars().next() {
                                let line_content = if text.len() > 1 { &text[1..] } else { "" };
                                match first_char {
                                    '+' => add_lines.push(line_content.to_string()),
                                    '-' => remove_lines.push(line_content.to_string()),
                                    ' ' => context_lines.push(line_content.to_string()),
                                    _ => {}
                                }
                            }
                        }
                        Rule::eof_line => {}
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    Ok(PatchHunk::Update {
        path,
        new_path,
        context: if context_lines.is_empty() {
            None
        } else {
            Some(context_lines.join("\n"))
        },
        remove: if remove_lines.is_empty() {
            None
        } else {
            Some(remove_lines.join("\n"))
        },
        add: add_lines.join("\n"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_add_file() {
        let input = r#"*** Begin Patch
*** Add File: test.txt
+line 1
+line 2
*** End Patch
"#;
        let patch = parse_freeform_patch(input).expect("should parse");
        assert_eq!(patch.hunks.len(), 1);
        match &patch.hunks[0] {
            PatchHunk::Add { path, content } => {
                assert_eq!(path, "test.txt");
                assert!(content.contains("line 1"));
            }
            _ => panic!("expected Add hunk"),
        }
    }

    #[test]
    fn parse_delete_file() {
        let input = r#"*** Begin Patch
*** Delete File: old.txt
*** End Patch
"#;
        let patch = parse_freeform_patch(input).expect("should parse");
        assert_eq!(patch.hunks.len(), 1);
        match &patch.hunks[0] {
            PatchHunk::Delete { path } => {
                assert_eq!(path, "old.txt");
            }
            _ => panic!("expected Delete hunk"),
        }
    }
}
