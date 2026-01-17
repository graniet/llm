const SSE_DELIMITER: &str = "\n\n";

pub(super) struct SseEventBuffer {
    buffer: String,
    utf8_buffer: Vec<u8>,
}

impl SseEventBuffer {
    pub(super) fn new() -> Self {
        Self {
            buffer: String::new(),
            utf8_buffer: Vec::new(),
        }
    }

    pub(super) fn push_bytes(&mut self, bytes: &[u8]) {
        self.utf8_buffer.extend_from_slice(bytes);
        match std::str::from_utf8(&self.utf8_buffer) {
            Ok(text) => {
                self.buffer.push_str(text);
                self.utf8_buffer.clear();
            }
            Err(err) => self.consume_valid_prefix(err.valid_up_to()),
        }
    }

    pub(super) fn drain_events(&mut self) -> Vec<String> {
        let mut events = Vec::new();
        while let Some(event) = self.next_event() {
            events.push(event);
        }
        events
    }

    fn consume_valid_prefix(&mut self, valid_up_to: usize) {
        if valid_up_to == 0 {
            return;
        }

        let valid = String::from_utf8_lossy(&self.utf8_buffer[..valid_up_to]);
        self.buffer.push_str(&valid);
        self.utf8_buffer.drain(..valid_up_to);
    }

    fn next_event(&mut self) -> Option<String> {
        let pos = self.buffer.find(SSE_DELIMITER)?;
        let end = pos + SSE_DELIMITER.len();
        let event = self.buffer[..end].to_string();
        self.buffer.drain(..end);
        Some(event)
    }
}
