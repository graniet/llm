//! Shell script safety checks.

use super::is_safe_command;

/// Check if a shell -c or -lc command is safe.
pub fn is_safe_shell_command(binary: &str, command: &[String]) -> bool {
    if !matches!(binary, "bash" | "sh") {
        return false;
    }

    // Look for -c or -lc flag
    let script_idx = command.iter().position(|arg| arg == "-c" || arg == "-lc");
    let Some(idx) = script_idx else {
        return false;
    };

    // The script should be the next argument
    let Some(script) = command.get(idx + 1) else {
        return false;
    };

    // Parse and check each command in the script
    is_safe_shell_script(script)
}

/// Check if a shell script contains only safe commands.
fn is_safe_shell_script(script: &str) -> bool {
    // Split by safe operators: &&, ||, ;, |
    // Reject if we see unsafe operators: >, <, >>, $(, `, (, {
    if script.contains('>')
        || script.contains('<')
        || script.contains("$(")
        || script.contains('`')
        || script.contains('(')
        || script.contains('{')
    {
        return false;
    }

    // Split by safe operators and check each command
    let commands: Vec<&str> = script
        .split("&&")
        .flat_map(|s| s.split("||"))
        .flat_map(|s| s.split(';'))
        .flat_map(|s| s.split('|'))
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();

    if commands.is_empty() {
        return false;
    }

    commands.iter().all(|cmd| {
        let parts: Vec<String> = shell_words::split(cmd).unwrap_or_default();
        if parts.is_empty() {
            return false;
        }
        is_safe_command(&parts)
    })
}
