//! Tool mutation detection for parallel execution.

/// Tools that are known to be read-only (non-mutating).
const NON_MUTATING_TOOLS: &[&str] = &[
    "echo",
    "time_now",
    "file_read",
    "search",
    "ls",
    "rollback", // summary action is non-mutating
];

/// Tools that are known to be write/mutating.
const MUTATING_TOOLS: &[&str] = &["shell", "shell_write", "patch", "plan"];

/// Check if a tool is mutating (writes to filesystem or has side effects).
pub fn is_mutating_tool(tool_name: &str) -> bool {
    if MUTATING_TOOLS.contains(&tool_name) {
        return true;
    }
    if NON_MUTATING_TOOLS.contains(&tool_name) {
        return false;
    }
    // Unknown tools are considered mutating for safety
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_mutating_tool() {
        assert!(!is_mutating_tool("file_read"));
        assert!(!is_mutating_tool("search"));
        assert!(!is_mutating_tool("ls"));
        assert!(!is_mutating_tool("echo"));

        assert!(is_mutating_tool("shell"));
        assert!(is_mutating_tool("shell_write"));
        assert!(is_mutating_tool("patch"));
        assert!(is_mutating_tool("plan"));

        // Unknown tools are mutating
        assert!(is_mutating_tool("unknown_tool"));
    }
}
