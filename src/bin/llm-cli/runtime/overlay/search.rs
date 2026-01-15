use crate::conversation::MessageId;

#[derive(Debug, Clone)]
pub struct SearchState {
    pub query: String,
    pub matches: Vec<MessageId>,
    pub selected: usize,
}

impl SearchState {
    pub fn new() -> Self {
        Self {
            query: String::new(),
            matches: Vec::new(),
            selected: 0,
        }
    }

    pub fn push_query(&mut self, ch: char) {
        self.query.push(ch);
    }

    pub fn pop_query(&mut self) {
        self.query.pop();
    }

    pub fn next(&mut self) {
        if !self.matches.is_empty() {
            self.selected = (self.selected + 1).min(self.matches.len() - 1);
        }
    }

    pub fn prev(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
        }
    }
}
