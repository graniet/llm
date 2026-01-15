#[derive(Debug, Default, Clone, Copy)]
pub struct ScrollState {
    offset: u16,
}

impl ScrollState {
    pub fn offset(&self) -> u16 {
        self.offset
    }

    pub fn reset(&mut self) {
        self.offset = 0;
    }

    pub fn scroll_up(&mut self, lines: u16) {
        self.offset = self.offset.saturating_add(lines);
    }

    pub fn scroll_down(&mut self, lines: u16) {
        self.offset = self.offset.saturating_sub(lines);
    }
}
