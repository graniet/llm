#[derive(Debug, Clone)]
pub struct PagerState {
    pub title: String,
    pub lines: Vec<String>,
    pub scroll: u16,
}

impl PagerState {
    pub fn new(title: impl Into<String>, content: &str) -> Self {
        let lines = content.lines().map(|line| line.to_string()).collect();
        Self {
            title: title.into(),
            lines,
            scroll: 0,
        }
    }

    pub fn max_scroll(&self, height: u16) -> u16 {
        let total = self.lines.len() as u16;
        total.saturating_sub(height)
    }

    pub fn scroll_up(&mut self, lines: u16) {
        self.scroll = self.scroll.saturating_sub(lines);
    }

    pub fn scroll_down(&mut self, lines: u16, height: u16) {
        let max = self.max_scroll(height);
        self.scroll = (self.scroll + lines).min(max);
    }

    pub fn scroll_top(&mut self) {
        self.scroll = 0;
    }

    pub fn scroll_bottom(&mut self, height: u16) {
        self.scroll = self.max_scroll(height);
    }
}
