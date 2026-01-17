//! Safe command detection for auto-approval.
//!
//! Commands in this list are considered safe because they don't modify state.

#![allow(dead_code)]

mod shell_safety;
mod tool_safety;

use shell_safety::is_safe_shell_command;
use tool_safety::{is_safe_find, is_safe_git, is_safe_ripgrep, is_safe_sed};

/// Check if a command is known to be safe (read-only, no side effects).
///
/// Safe commands can be auto-approved without user confirmation.
#[must_use]
pub fn is_safe_command(command: &[String]) -> bool {
    let Some(cmd) = command.first() else {
        return false;
    };

    // Extract binary name from path
    let binary = std::path::Path::new(cmd)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(cmd);

    // Normalize zsh to bash for checking
    let binary = if binary == "zsh" { "bash" } else { binary };

    // Check if it's a direct safe command
    if is_safe_binary(binary, command) {
        return true;
    }

    // Check bash -c or bash -lc patterns
    if is_safe_shell_command(binary, command) {
        return true;
    }

    false
}

/// Check if the binary itself is safe with its arguments.
fn is_safe_binary(binary: &str, command: &[String]) -> bool {
    match binary {
        // Always safe read-only commands
        "cat" | "cd" | "cut" | "echo" | "expr" | "false" | "head" | "id" | "ls" | "nl"
        | "paste" | "pwd" | "rev" | "seq" | "stat" | "tail" | "tr" | "true" | "uname" | "uniq"
        | "wc" | "which" | "whoami" | "hostname" | "date" | "env" | "printenv" | "file"
        | "type" | "basename" | "dirname" | "realpath" | "readlink" => true,

        // grep is safe
        "grep" | "egrep" | "fgrep" => true,

        // find needs argument checking
        "find" => is_safe_find(command),

        // ripgrep needs argument checking
        "rg" => is_safe_ripgrep(command),

        // git with safe subcommands only
        "git" => is_safe_git(command),

        // cargo check is safe
        "cargo" => command.get(1).is_some_and(|sub| sub == "check"),

        // sed with -n and print pattern only
        "sed" => is_safe_sed(command),

        // sort without output redirection
        "sort" => !command.iter().any(|a| a.starts_with("-o")),

        // diff is safe
        "diff" => true,

        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cmd(args: &[&str]) -> Vec<String> {
        args.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn basic_safe_commands() {
        assert!(is_safe_command(&cmd(&["ls"])));
        assert!(is_safe_command(&cmd(&["ls", "-la"])));
        assert!(is_safe_command(&cmd(&["cat", "file.txt"])));
        assert!(is_safe_command(&cmd(&["pwd"])));
        assert!(is_safe_command(&cmd(&["whoami"])));
        assert!(is_safe_command(&cmd(&["echo", "hello"])));
    }

    #[test]
    fn git_safe_subcommands() {
        assert!(is_safe_command(&cmd(&["git", "status"])));
        assert!(is_safe_command(&cmd(&["git", "log"])));
        assert!(is_safe_command(&cmd(&["git", "diff"])));
        assert!(!is_safe_command(&cmd(&["git", "push"])));
        assert!(!is_safe_command(&cmd(&["git", "commit"])));
    }

    #[test]
    fn find_safety() {
        assert!(is_safe_command(&cmd(&["find", ".", "-name", "*.rs"])));
        assert!(!is_safe_command(&cmd(&["find", ".", "-delete"])));
        assert!(!is_safe_command(&cmd(&[
            "find", ".", "-exec", "rm", "{}", ";"
        ])));
    }

    #[test]
    fn ripgrep_safety() {
        assert!(is_safe_command(&cmd(&["rg", "pattern", "-n"])));
        assert!(!is_safe_command(&cmd(&["rg", "--pre", "cmd", "pattern"])));
        assert!(!is_safe_command(&cmd(&["rg", "-z", "pattern"])));
    }

    #[test]
    fn shell_script_safety() {
        assert!(is_safe_command(&cmd(&["bash", "-c", "ls && pwd"])));
        assert!(is_safe_command(&cmd(&["bash", "-lc", "git status"])));
        assert!(!is_safe_command(&cmd(&["bash", "-c", "rm -rf /"])));
        assert!(!is_safe_command(&cmd(&["bash", "-c", "ls > out.txt"])));
    }

    #[test]
    fn sed_safety() {
        assert!(is_safe_command(&cmd(&["sed", "-n", "1,5p", "file.txt"])));
        assert!(is_safe_command(&cmd(&["sed", "-n", "10p"])));
        assert!(!is_safe_command(&cmd(&["sed", "-i", "s/a/b/g"])));
    }

    #[test]
    fn unsafe_commands() {
        assert!(!is_safe_command(&cmd(&["rm", "file"])));
        assert!(!is_safe_command(&cmd(&["mv", "a", "b"])));
        assert!(!is_safe_command(&cmd(&["chmod", "+x", "file"])));
        assert!(!is_safe_command(&cmd(&["curl", "url"])));
    }
}
