use crate::terminal::{AnimationLevel, TerminalCapabilities};

#[derive(Debug, Clone)]
pub struct AnimationState {
    frame: u64,
    slow_render: bool,
}

impl AnimationState {
    pub fn new() -> Self {
        Self {
            frame: 0,
            slow_render: false,
        }
    }

    pub fn tick(&mut self, caps: &TerminalCapabilities) -> bool {
        if self.level(caps) == AnimationLevel::Static {
            return false;
        }
        self.frame = self.frame.wrapping_add(1);
        true
    }

    pub fn frame(&self) -> u64 {
        self.frame
    }

    pub fn disable(&mut self) {
        self.slow_render = true;
    }

    pub fn level(&self, caps: &TerminalCapabilities) -> AnimationLevel {
        if self.slow_render {
            return AnimationLevel::Static;
        }
        caps.animation_level()
    }
}

impl Default for AnimationState {
    fn default() -> Self {
        Self::new()
    }
}
