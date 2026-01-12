mod ansi;
mod lab;

use ratatui::style::Color;

use super::capabilities::ColorLevel;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rgb {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Rgb {
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }
}

#[derive(Debug, Clone)]
pub struct TerminalPalette {
    level: ColorLevel,
}

impl TerminalPalette {
    pub fn new(level: ColorLevel) -> Self {
        Self { level }
    }

    pub fn map(&self, rgb: Rgb) -> Color {
        match self.level {
            ColorLevel::TrueColor => Color::Rgb(rgb.r, rgb.g, rgb.b),
            ColorLevel::Ansi256 => Color::Indexed(ansi::nearest_index(rgb, 256)),
            ColorLevel::Ansi16 => Color::Indexed(ansi::nearest_index(rgb, 16)),
            ColorLevel::Ansi8 => Color::Indexed(ansi::nearest_index(rgb, 8)),
            ColorLevel::None => Color::Reset,
        }
    }

    pub fn blend(&self, base: Rgb, highlight: Rgb, t: f32) -> Color {
        let blended = lab::blend_lab(base, highlight, t);
        self.map(blended)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_truecolor_exact() {
        let palette = TerminalPalette::new(ColorLevel::TrueColor);
        let mapped = palette.map(Rgb::new(10, 20, 30));
        assert_eq!(mapped, Color::Rgb(10, 20, 30));
    }

    #[test]
    fn maps_ansi256_within_bounds() {
        let palette = TerminalPalette::new(ColorLevel::Ansi256);
        let mapped = palette.map(Rgb::new(120, 130, 140));
        match mapped {
            Color::Indexed(idx) => assert!(idx <= ansi::ANSI_256_MAX),
            _ => panic!("expected indexed color"),
        }
    }
}
