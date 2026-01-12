mod capabilities;
mod palette;

#[cfg(test)]
pub use capabilities::ColorLevel;
pub use capabilities::{AnimationLevel, TerminalCapabilities};
pub use palette::{Rgb, TerminalPalette};
