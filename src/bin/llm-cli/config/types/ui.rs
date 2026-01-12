use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Deserialize, Serialize, Eq, PartialEq, Default)]
#[serde(rename_all = "lowercase")]
pub enum NavigationMode {
    #[default]
    Simple,
    Vi,
}

/// UI configuration for the TUI
///
/// # Available Themes
/// - `warm` (default): Orange/terracotta tones inspired by Anthropic/Claude
/// - `cool`: Blue/cyan tones for those who prefer cooler colors
/// - `mono`: High contrast monochrome for accessibility
/// - `codex`: Legacy alias for `warm` theme
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct UiConfig {
    /// Theme name: "warm" (default), "cool", "mono", or "codex" (legacy)
    pub theme: String,
    /// Show timestamps on messages
    pub timestamps: bool,
    /// Wrap code blocks
    pub wrap_code: bool,
    /// Navigation mode: "simple" (arrow keys) or "vi" (vim keybindings)
    pub navigation_mode: NavigationMode,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            theme: "warm".to_string(),
            timestamps: true,
            wrap_code: false,
            navigation_mode: NavigationMode::Simple,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_theme_is_warm() {
        let config = UiConfig::default();
        assert_eq!(config.theme, "warm");
    }

    #[test]
    fn default_navigation_is_simple() {
        let config = UiConfig::default();
        assert_eq!(config.navigation_mode, NavigationMode::Simple);
    }

    #[test]
    fn deserialize_theme() {
        let json = r#"{"theme": "cool"}"#;
        let config: UiConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.theme, "cool");
    }

    #[test]
    fn deserialize_vi_mode() {
        let json = r#"{"navigation_mode": "vi"}"#;
        let config: UiConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.navigation_mode, NavigationMode::Vi);
    }
}
