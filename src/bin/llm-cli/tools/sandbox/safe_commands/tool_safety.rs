//! Safety checks for individual tools.

/// Check if find command is safe (no exec, delete, or write options).
pub fn is_safe_find(command: &[String]) -> bool {
    const UNSAFE_OPTIONS: &[&str] = &[
        "-exec", "-execdir", "-ok", "-okdir", "-delete", "-fls", "-fprint", "-fprint0", "-fprintf",
    ];

    !command
        .iter()
        .any(|arg| UNSAFE_OPTIONS.contains(&arg.as_str()))
}

/// Check if ripgrep command is safe.
pub fn is_safe_ripgrep(command: &[String]) -> bool {
    const UNSAFE_WITH_ARGS: &[&str] = &["--pre", "--hostname-bin"];
    const UNSAFE_FLAGS: &[&str] = &["--search-zip", "-z"];

    !command.iter().any(|arg| {
        UNSAFE_FLAGS.contains(&arg.as_str())
            || UNSAFE_WITH_ARGS
                .iter()
                .any(|&opt| arg == opt || arg.starts_with(&format!("{opt}=")))
    })
}

/// Check if git command is safe (read-only subcommands only).
pub fn is_safe_git(command: &[String]) -> bool {
    const SAFE_SUBCOMMANDS: &[&str] = &[
        "branch",
        "status",
        "log",
        "diff",
        "show",
        "ls-files",
        "ls-tree",
        "rev-parse",
        "describe",
        "tag",
        "remote",
        "config",
    ];

    command
        .get(1)
        .is_some_and(|sub| SAFE_SUBCOMMANDS.contains(&sub.as_str()))
}

/// Check if sed command is safe (only -n with print pattern).
pub fn is_safe_sed(command: &[String]) -> bool {
    // Only allow: sed -n <pattern>p [file]
    if command.len() < 3 || command.len() > 4 {
        return false;
    }

    if command.get(1).is_none_or(|a| a != "-n") {
        return false;
    }

    // Pattern must end with 'p' and be numeric ranges
    command.get(2).is_some_and(|p| is_valid_sed_pattern(p))
}

/// Check if sed pattern is valid (numeric ranges only).
fn is_valid_sed_pattern(pattern: &str) -> bool {
    let Some(core) = pattern.strip_suffix('p') else {
        return false;
    };

    let parts: Vec<&str> = core.split(',').collect();
    match parts.as_slice() {
        [num] => !num.is_empty() && num.chars().all(|c| c.is_ascii_digit()),
        [a, b] => {
            !a.is_empty()
                && !b.is_empty()
                && a.chars().all(|c| c.is_ascii_digit())
                && b.chars().all(|c| c.is_ascii_digit())
        }
        _ => false,
    }
}
