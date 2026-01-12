use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

use super::wrap::wrap_ranges;

#[derive(Debug, Default, Clone)]
pub struct InputBuffer {
    text: String,
    cursor: usize,
}

impl InputBuffer {
    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    pub fn set_text(&mut self, value: String) {
        self.text = value;
        self.cursor = self.cursor.min(self.text.len());
    }

    pub fn take_text(&mut self) -> String {
        self.cursor = 0;
        std::mem::take(&mut self.text)
    }

    pub fn insert_str(&mut self, value: &str) {
        let pos = self.cursor.min(self.text.len());
        self.text.insert_str(pos, value);
        self.cursor = pos + value.len();
    }

    pub fn insert_char(&mut self, ch: char) {
        let pos = self.cursor.min(self.text.len());
        self.text.insert(pos, ch);
        self.cursor = pos + ch.len_utf8();
    }

    pub fn backspace(&mut self) {
        if let Some(prev) = self.prev_grapheme_start() {
            self.text.replace_range(prev..self.cursor, "");
            self.cursor = prev;
        }
    }

    pub fn delete(&mut self) {
        let next = self.next_grapheme_start();
        if let Some(next_idx) = next {
            self.text.replace_range(self.cursor..next_idx, "");
        }
    }

    pub fn move_left(&mut self) {
        if let Some(prev) = self.prev_grapheme_start() {
            self.cursor = prev;
        }
    }

    pub fn move_right(&mut self) {
        if let Some(next) = self.next_grapheme_start() {
            self.cursor = next;
        }
    }

    pub fn move_up(&mut self, width: u16) {
        let (row, col) = self.cursor_position(width);
        if row == 0 {
            return;
        }
        let ranges = wrap_ranges(&self.text, width);
        if let Some(range) = ranges.get(row.saturating_sub(1) as usize) {
            self.cursor = cursor_at_column(&self.text, range.start, range.end, col);
        }
    }

    pub fn move_down(&mut self, width: u16) {
        let (row, col) = self.cursor_position(width);
        let ranges = wrap_ranges(&self.text, width);
        if row as usize + 1 >= ranges.len() {
            return;
        }
        if let Some(range) = ranges.get(row as usize + 1) {
            self.cursor = cursor_at_column(&self.text, range.start, range.end, col);
        }
    }

    pub fn move_home(&mut self) {
        let start = self.text[..self.cursor]
            .rfind('\n')
            .map(|idx| idx + 1)
            .unwrap_or(0);
        self.cursor = start;
    }

    pub fn move_end(&mut self) {
        let end = self.text[self.cursor..]
            .find('\n')
            .map(|idx| self.cursor + idx)
            .unwrap_or(self.text.len());
        self.cursor = end;
    }

    pub fn newline(&mut self) {
        self.insert_char('\n');
    }

    pub fn wrapped_lines(&self, width: u16) -> Vec<String> {
        wrap_ranges(&self.text, width)
            .into_iter()
            .map(|range| self.text[range].to_string())
            .collect()
    }

    pub fn cursor_position(&self, width: u16) -> (u16, u16) {
        let ranges = wrap_ranges(&self.text, width);
        for (row, range) in ranges.iter().enumerate() {
            if self.cursor >= range.start && self.cursor <= range.end {
                let slice = &self.text[range.start..self.cursor];
                let col = slice.width() as u16;
                return (row as u16, col);
            }
        }
        (0, 0)
    }

    fn prev_grapheme_start(&self) -> Option<usize> {
        if self.cursor == 0 {
            return None;
        }
        self.text[..self.cursor]
            .grapheme_indices(true)
            .next_back()
            .map(|(idx, _)| idx)
    }

    fn next_grapheme_start(&self) -> Option<usize> {
        if self.cursor >= self.text.len() {
            return None;
        }
        let remaining = &self.text[self.cursor..];
        remaining
            .grapheme_indices(true)
            .nth(1)
            .map(|(idx, _)| self.cursor + idx)
            .or(Some(self.text.len()))
    }
}

fn cursor_at_column(text: &str, start: usize, end: usize, target_col: u16) -> usize {
    let mut width = 0usize;
    let slice = &text[start..end];
    for (idx, grapheme) in slice.grapheme_indices(true) {
        width += grapheme.width();
        if width as u16 >= target_col {
            return start + idx + grapheme.len();
        }
    }
    end
}
