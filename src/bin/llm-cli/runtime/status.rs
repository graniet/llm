use std::time::Instant;

#[derive(Debug, Clone)]
pub enum AppStatus {
    Idle,
    Thinking,
    Streaming,
    Error(String),
}

impl AppStatus {
    pub fn is_busy(&self) -> bool {
        matches!(self, AppStatus::Thinking | AppStatus::Streaming)
    }
}

#[derive(Debug, Clone, Default)]
pub struct StatusMetrics {
    started_at: Option<Instant>,
    tokens: Option<u32>,
    last_reported_sec: Option<u64>,
}

impl StatusMetrics {
    pub fn start(&mut self) {
        if self.started_at.is_none() {
            self.started_at = Some(Instant::now());
        }
        self.tokens = None;
        self.last_reported_sec = None;
    }

    pub fn stop(&mut self) {
        self.started_at = None;
        self.tokens = None;
        self.last_reported_sec = None;
    }

    pub fn update_tokens(&mut self, tokens: u32) {
        self.tokens = Some(tokens);
    }

    pub fn elapsed_ms(&self) -> Option<u128> {
        self.started_at.map(|at| at.elapsed().as_millis())
    }

    pub fn tokens(&self) -> Option<u32> {
        self.tokens
    }

    pub fn should_redraw(&mut self) -> bool {
        let Some(started_at) = self.started_at else {
            return false;
        };
        let secs = started_at.elapsed().as_secs();
        if self.last_reported_sec == Some(secs) {
            return false;
        }
        self.last_reported_sec = Some(secs);
        true
    }
}
