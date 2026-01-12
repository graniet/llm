use std::time::{Duration, Instant};

const BURST_WINDOW: Duration = Duration::from_millis(24);
const BURST_HOLD: Duration = Duration::from_millis(180);
const BURST_THRESHOLD: usize = 5;

#[derive(Debug, Default, Clone)]
pub struct PasteDetector {
    last_at: Option<Instant>,
    burst_len: usize,
    burst_until: Option<Instant>,
}

impl PasteDetector {
    pub fn record_char(&mut self, now: Instant) -> bool {
        let within_window = self
            .last_at
            .map(|at| now.saturating_duration_since(at) <= BURST_WINDOW)
            .unwrap_or(false);
        if within_window {
            self.burst_len = self.burst_len.saturating_add(1);
        } else {
            self.burst_len = 1;
        }
        self.last_at = Some(now);
        if self.burst_len >= BURST_THRESHOLD {
            self.burst_until = Some(now + BURST_HOLD);
        }
        self.is_burst(now)
    }

    pub fn record_paste(&mut self, now: Instant) {
        self.last_at = Some(now);
        self.burst_len = 0;
        self.burst_until = Some(now + BURST_HOLD);
    }

    pub fn is_burst(&self, now: Instant) -> bool {
        self.burst_until.map(|until| now <= until).unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_burst_after_threshold() {
        let mut detector = PasteDetector::default();
        let start = Instant::now();
        for idx in 0..BURST_THRESHOLD {
            let now = start + BURST_WINDOW / 2 * idx as u32;
            detector.record_char(now);
        }
        assert!(detector.is_burst(start + BURST_WINDOW));
    }
}
