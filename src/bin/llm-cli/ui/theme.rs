use ratatui::style::{Color, Modifier, Style};

use crate::terminal::{Rgb, TerminalPalette};

/// Unicode box drawing characters for borders
#[allow(dead_code)]
pub mod borders {
    pub const LEFT_BORDER: &str = "▎";
}

/// Unicode indicators for status and UI elements
pub mod indicators {
    pub const PROMPT: &str = "❯";
    pub const BULLET: &str = "●";
    pub const CHECK: &str = "✓";
    pub const CROSS: &str = "✗";
    pub const EXPAND: &str = "▸";
    pub const COLLAPSE: &str = "▾";
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Theme {
    // Message styles with left border colors
    pub user_border: Style,
    pub user_bg: Style,
    pub assistant_border: Style,
    pub assistant_bg: Style,
    pub tool_border: Style,
    pub tool_bg: Style,
    // Legacy compatibility
    pub user: Style,
    pub assistant: Style,
    pub tool: Style,
    pub tool_dim: Style,
    // Error and status
    pub error: Style,
    pub error_border: Style,
    // Diff styles
    pub diff_add: Style,
    pub diff_remove: Style,
    pub diff_header: Style,
    pub diff_lineno: Style,
    // Status bar
    pub status: Style,
    pub status_ok: Style,
    pub status_warn: Style,
    pub status_error: Style,
    pub status_indicator: Style,
    // UI elements
    pub accent: Style,
    pub muted: Style,
    pub border: Style,
    pub border_focused: Style,
    // Code blocks
    pub code_bg: Style,
    pub code_border: Style,
    pub code_header: Style,
    // Input
    pub prompt: Style,
    pub mode_indicator: Style,
}

impl Theme {
    pub fn from_name(name: &str, palette: &TerminalPalette) -> Self {
        match name.to_lowercase().as_str() {
            "mono" => Self::mono_with_palette(palette),
            "cool" => Self::cool_with_palette(palette),
            _ => Self::warm_with_palette(palette),
        }
    }

    /// Warm theme - Anthropic/Claude inspired with orange/terracotta tones
    pub fn warm_with_palette(palette: &TerminalPalette) -> Self {
        // Warm color palette
        let orange = Rgb::new(217, 119, 87); // #D97757 - Anthropic orange
        let terracotta = Rgb::new(198, 120, 95); // #C6785F - Terracotta
        let coral = Rgb::new(230, 145, 120); // #E69178 - Coral accent
        let _sand = Rgb::new(245, 235, 220); // #F5EBDC - Light sand (reserved for future use)

        // Background tones
        let user_bg_color = Rgb::new(45, 35, 30); // Dark warm brown
        let assistant_bg_color = Rgb::new(30, 32, 35); // Neutral dark
        let tool_bg_color = Rgb::new(35, 40, 38); // Dark teal-gray
        let code_bg_color = Rgb::new(35, 35, 40); // Dark blue-gray

        // Accent colors
        let teal = Rgb::new(94, 180, 160); // #5EB4A0 - Teal for tools
        let green = Rgb::new(120, 200, 140); // #78C88C - Success green
        let yellow = Rgb::new(233, 182, 89); // #E9B659 - Warning yellow
        let muted_gray = Rgb::new(140, 135, 130); // Warm gray
        let dim_gray = Rgb::new(100, 95, 90); // Dimmer warm gray

        Self {
            // User messages - orange left border
            user_border: Style::default().fg(palette.map(orange)),
            user_bg: Style::default().bg(palette.map(user_bg_color)),
            // Assistant messages - coral left border
            assistant_border: Style::default().fg(palette.map(coral)),
            assistant_bg: Style::default().bg(palette.map(assistant_bg_color)),
            // Tool messages - teal left border
            tool_border: Style::default().fg(palette.map(teal)),
            tool_bg: Style::default().bg(palette.map(tool_bg_color)),
            // Legacy - keep background style for compatibility
            user: Style::default().bg(palette.map(user_bg_color)),
            assistant: Style::default(),
            tool: Style::default().fg(palette.map(teal)),
            tool_dim: Style::default()
                .fg(palette.map(teal))
                .add_modifier(Modifier::DIM),
            // Errors
            error: Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            error_border: Style::default().fg(Color::Red),
            // Diffs
            diff_add: Style::default().fg(palette.map(green)),
            diff_remove: Style::default().fg(Color::Red),
            diff_header: Style::default()
                .fg(palette.map(muted_gray))
                .add_modifier(Modifier::DIM),
            diff_lineno: Style::default()
                .fg(palette.map(dim_gray))
                .add_modifier(Modifier::DIM),
            // Status
            status: Style::default()
                .fg(palette.map(dim_gray))
                .add_modifier(Modifier::DIM),
            status_ok: Style::default().fg(palette.map(green)),
            status_warn: Style::default().fg(palette.map(yellow)),
            status_error: Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            status_indicator: Style::default().fg(palette.map(orange)),
            // UI elements
            accent: Style::default()
                .fg(palette.map(orange))
                .add_modifier(Modifier::BOLD),
            muted: Style::default().fg(palette.map(muted_gray)),
            border: Style::default().fg(palette.map(dim_gray)),
            border_focused: Style::default().fg(palette.map(orange)),
            // Code blocks
            code_bg: Style::default().bg(palette.map(code_bg_color)),
            code_border: Style::default().fg(palette.map(dim_gray)),
            code_header: Style::default()
                .fg(palette.map(muted_gray))
                .add_modifier(Modifier::DIM),
            // Input
            prompt: Style::default()
                .fg(palette.map(orange))
                .add_modifier(Modifier::BOLD),
            mode_indicator: Style::default().fg(palette.map(terracotta)),
        }
    }

    /// Cool theme - Blue/cyan tones for those who prefer cooler colors
    pub fn cool_with_palette(palette: &TerminalPalette) -> Self {
        // Cool color palette
        let blue = Rgb::new(100, 150, 220); // #6496DC - Primary blue
        let cyan = Rgb::new(80, 180, 200); // #50B4C8 - Cyan accent
        let ice = Rgb::new(140, 190, 230); // #8CBEE6 - Ice blue

        // Background tones
        let user_bg_color = Rgb::new(25, 35, 50); // Dark blue
        let assistant_bg_color = Rgb::new(30, 32, 38); // Neutral dark blue
        let tool_bg_color = Rgb::new(30, 40, 45); // Dark cyan-gray
        let code_bg_color = Rgb::new(30, 35, 45); // Dark blue-gray

        // Accent colors
        let teal = Rgb::new(94, 209, 170); // Original teal for tools
        let green = Rgb::new(96, 200, 136); // Success green
        let yellow = Rgb::new(233, 182, 89); // Warning yellow
        let muted_gray = Rgb::new(130, 140, 150); // Cool gray
        let dim_gray = Rgb::new(90, 100, 110); // Dimmer cool gray

        Self {
            user_border: Style::default().fg(palette.map(blue)),
            user_bg: Style::default().bg(palette.map(user_bg_color)),
            assistant_border: Style::default().fg(palette.map(ice)),
            assistant_bg: Style::default().bg(palette.map(assistant_bg_color)),
            tool_border: Style::default().fg(palette.map(teal)),
            tool_bg: Style::default().bg(palette.map(tool_bg_color)),
            user: Style::default().bg(palette.map(user_bg_color)),
            assistant: Style::default(),
            tool: Style::default().fg(palette.map(teal)),
            tool_dim: Style::default()
                .fg(palette.map(teal))
                .add_modifier(Modifier::DIM),
            error: Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            error_border: Style::default().fg(Color::Red),
            diff_add: Style::default().fg(palette.map(green)),
            diff_remove: Style::default().fg(Color::Red),
            diff_header: Style::default()
                .fg(palette.map(muted_gray))
                .add_modifier(Modifier::DIM),
            diff_lineno: Style::default()
                .fg(palette.map(dim_gray))
                .add_modifier(Modifier::DIM),
            status: Style::default()
                .fg(palette.map(dim_gray))
                .add_modifier(Modifier::DIM),
            status_ok: Style::default().fg(palette.map(green)),
            status_warn: Style::default().fg(palette.map(yellow)),
            status_error: Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            status_indicator: Style::default().fg(palette.map(cyan)),
            accent: Style::default()
                .fg(palette.map(blue))
                .add_modifier(Modifier::BOLD),
            muted: Style::default().fg(palette.map(muted_gray)),
            border: Style::default().fg(palette.map(dim_gray)),
            border_focused: Style::default().fg(palette.map(blue)),
            code_bg: Style::default().bg(palette.map(code_bg_color)),
            code_border: Style::default().fg(palette.map(dim_gray)),
            code_header: Style::default()
                .fg(palette.map(muted_gray))
                .add_modifier(Modifier::DIM),
            prompt: Style::default()
                .fg(palette.map(cyan))
                .add_modifier(Modifier::BOLD),
            mode_indicator: Style::default().fg(palette.map(ice)),
        }
    }

    /// Mono theme - High contrast monochrome for accessibility
    fn mono_with_palette(palette: &TerminalPalette) -> Self {
        let white = Rgb::new(220, 220, 220);
        let light_gray = Rgb::new(180, 180, 180);
        let mid_gray = Rgb::new(140, 140, 140);
        let dark_gray = Rgb::new(80, 80, 80);
        let darker_gray = Rgb::new(50, 50, 50);
        let base_bg = Rgb::new(35, 35, 35);

        Self {
            user_border: Style::default().fg(palette.map(white)),
            user_bg: Style::default().bg(palette.map(base_bg)),
            assistant_border: Style::default().fg(palette.map(light_gray)),
            assistant_bg: Style::default(),
            tool_border: Style::default().fg(palette.map(mid_gray)),
            tool_bg: Style::default().bg(palette.map(darker_gray)),
            user: Style::default().bg(palette.map(base_bg)),
            assistant: Style::default(),
            tool: Style::default().fg(palette.map(light_gray)),
            tool_dim: Style::default()
                .fg(palette.map(light_gray))
                .add_modifier(Modifier::DIM),
            error: Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            error_border: Style::default().fg(Color::Red),
            diff_add: Style::default().fg(palette.map(light_gray)),
            diff_remove: Style::default().fg(Color::Red),
            diff_header: Style::default()
                .fg(palette.map(mid_gray))
                .add_modifier(Modifier::DIM),
            diff_lineno: Style::default()
                .fg(palette.map(dark_gray))
                .add_modifier(Modifier::DIM),
            status: Style::default()
                .fg(palette.map(dark_gray))
                .add_modifier(Modifier::DIM),
            status_ok: Style::default().fg(palette.map(mid_gray)),
            status_warn: Style::default().fg(Color::Yellow),
            status_error: Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            status_indicator: Style::default().fg(palette.map(white)),
            accent: Style::default()
                .fg(palette.map(white))
                .add_modifier(Modifier::BOLD),
            muted: Style::default().fg(palette.map(mid_gray)),
            border: Style::default().fg(palette.map(dark_gray)),
            border_focused: Style::default().fg(palette.map(white)),
            code_bg: Style::default().bg(palette.map(darker_gray)),
            code_border: Style::default().fg(palette.map(dark_gray)),
            code_header: Style::default()
                .fg(palette.map(mid_gray))
                .add_modifier(Modifier::DIM),
            prompt: Style::default()
                .fg(palette.map(white))
                .add_modifier(Modifier::BOLD),
            mode_indicator: Style::default().fg(palette.map(light_gray)),
        }
    }

    /// Legacy method for backward compatibility - maps to warm theme
    #[allow(dead_code)]
    pub fn codex_with_palette(palette: &TerminalPalette) -> Self {
        Self::warm_with_palette(palette)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terminal::ColorLevel;

    #[test]
    fn warm_theme_creates_valid_styles() {
        let palette = TerminalPalette::new(ColorLevel::TrueColor);
        let theme = Theme::warm_with_palette(&palette);
        // Verify key styles are not default
        assert_ne!(theme.user_border, Style::default());
        assert_ne!(theme.accent, Style::default());
    }

    #[test]
    fn cool_theme_creates_valid_styles() {
        let palette = TerminalPalette::new(ColorLevel::TrueColor);
        let theme = Theme::cool_with_palette(&palette);
        assert_ne!(theme.user_border, Style::default());
        assert_ne!(theme.accent, Style::default());
    }

    #[test]
    fn mono_theme_creates_valid_styles() {
        let palette = TerminalPalette::new(ColorLevel::TrueColor);
        let theme = Theme::mono_with_palette(&palette);
        assert_ne!(theme.user_border, Style::default());
    }

    #[test]
    fn from_name_selects_correct_theme() {
        let palette = TerminalPalette::new(ColorLevel::TrueColor);

        let warm = Theme::from_name("warm", &palette);
        let cool = Theme::from_name("cool", &palette);
        let mono = Theme::from_name("mono", &palette);
        let default = Theme::from_name("unknown", &palette);

        // warm and default should match
        assert_eq!(
            format!("{:?}", warm.accent),
            format!("{:?}", default.accent)
        );
        // cool should be different from warm
        assert_ne!(format!("{:?}", cool.accent), format!("{:?}", warm.accent));
        // mono should be different
        assert_ne!(format!("{:?}", mono.accent), format!("{:?}", warm.accent));
    }

    #[test]
    fn codex_maps_to_warm() {
        let palette = TerminalPalette::new(ColorLevel::TrueColor);
        let codex = Theme::codex_with_palette(&palette);
        let warm = Theme::warm_with_palette(&palette);
        assert_eq!(format!("{:?}", codex.accent), format!("{:?}", warm.accent));
    }
}
