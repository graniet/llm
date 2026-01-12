use std::env;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ColorLevel {
    TrueColor,
    Ansi256,
    Ansi16,
    Ansi8,
    None,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum AnimationLevel {
    Shimmer,
    Spinner,
    Static,
}

#[derive(Debug, Clone)]
pub struct TerminalCapabilities {
    pub color_level: ColorLevel,
    pub supports_mouse: bool,
    pub supports_bracketed_paste: bool,
    pub slow_terminal: bool,
}

impl TerminalCapabilities {
    pub fn detect() -> Self {
        let env = TerminalEnv::from_env();
        Self::from_env(env)
    }

    pub fn animation_level(&self) -> AnimationLevel {
        if self.slow_terminal || self.color_level == ColorLevel::None {
            return AnimationLevel::Static;
        }
        match self.color_level {
            ColorLevel::TrueColor => AnimationLevel::Shimmer,
            ColorLevel::Ansi256 | ColorLevel::Ansi16 => AnimationLevel::Spinner,
            ColorLevel::Ansi8 | ColorLevel::None => AnimationLevel::Static,
        }
    }

    fn from_env(env: TerminalEnv) -> Self {
        let color_level = detect_color_level(&env);
        let supports_mouse = supports_mouse(&env);
        let supports_bracketed_paste = supports_bracketed_paste(&env);
        let slow_terminal = is_slow_terminal(&env, color_level);
        Self {
            color_level,
            supports_mouse,
            supports_bracketed_paste,
            slow_terminal,
        }
    }
}

#[derive(Debug, Clone)]
struct TerminalEnv {
    term: Option<String>,
    colorterm: Option<String>,
    no_color: bool,
    ssh: bool,
}

impl TerminalEnv {
    fn from_env() -> Self {
        let term = env::var("TERM").ok();
        let colorterm = env::var("COLORTERM").ok();
        let no_color = env::var("NO_COLOR").is_ok();
        let ssh = env::var("SSH_CONNECTION").is_ok()
            || env::var("SSH_CLIENT").is_ok()
            || env::var("SSH_TTY").is_ok();
        Self {
            term,
            colorterm,
            no_color,
            ssh,
        }
    }
}

fn detect_color_level(env: &TerminalEnv) -> ColorLevel {
    if env.no_color {
        return ColorLevel::None;
    }
    let term = env.term.as_deref().unwrap_or_default();
    if term == "dumb" {
        return ColorLevel::None;
    }
    if has_truecolor(env.colorterm.as_deref()) {
        return ColorLevel::TrueColor;
    }
    if term.contains("256color") {
        return ColorLevel::Ansi256;
    }
    if term.contains("color") {
        return ColorLevel::Ansi16;
    }
    ColorLevel::Ansi8
}

fn has_truecolor(value: Option<&str>) -> bool {
    value
        .map(|v| v.to_lowercase())
        .map(|v| v.contains("truecolor") || v.contains("24bit"))
        .unwrap_or(false)
}

fn supports_mouse(env: &TerminalEnv) -> bool {
    let term = env.term.as_deref().unwrap_or_default();
    term != "dumb"
}

fn supports_bracketed_paste(env: &TerminalEnv) -> bool {
    let term = env.term.as_deref().unwrap_or_default();
    term != "dumb"
}

fn is_slow_terminal(env: &TerminalEnv, color_level: ColorLevel) -> bool {
    if !env.ssh {
        return false;
    }
    matches!(
        color_level,
        ColorLevel::Ansi8 | ColorLevel::Ansi16 | ColorLevel::None
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_with(term: Option<&str>, colorterm: Option<&str>, ssh: bool) -> TerminalEnv {
        TerminalEnv {
            term: term.map(|v| v.to_string()),
            colorterm: colorterm.map(|v| v.to_string()),
            no_color: false,
            ssh,
        }
    }

    #[test]
    fn detects_truecolor() {
        let env = env_with(Some("xterm-256color"), Some("truecolor"), false);
        let caps = TerminalCapabilities::from_env(env);
        assert_eq!(caps.color_level, ColorLevel::TrueColor);
        assert_eq!(caps.animation_level(), AnimationLevel::Shimmer);
    }

    #[test]
    fn detects_ansi256() {
        let env = env_with(Some("xterm-256color"), None, false);
        let caps = TerminalCapabilities::from_env(env);
        assert_eq!(caps.color_level, ColorLevel::Ansi256);
        assert_eq!(caps.animation_level(), AnimationLevel::Spinner);
    }

    #[test]
    fn ssh_limits_animation() {
        let env = env_with(Some("xterm-color"), None, true);
        let caps = TerminalCapabilities::from_env(env);
        assert_eq!(caps.color_level, ColorLevel::Ansi16);
        assert!(caps.slow_terminal);
        assert_eq!(caps.animation_level(), AnimationLevel::Static);
    }
}
