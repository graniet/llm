use ratatui::text::{Line, Text};
use textwrap::{Options, WordSplitter};

pub(super) fn wrapped_height(text: &Text<'_>, width: u16) -> u16 {
    if width == 0 {
        return 0;
    }
    let options = Options::new(width as usize)
        .word_splitter(WordSplitter::NoHyphenation)
        .break_words(false);
    let mut total = 0usize;
    for line in &text.lines {
        total = total.saturating_add(line_height(line, &options));
    }
    total.min(u16::MAX as usize) as u16
}

fn line_height(line: &Line<'_>, options: &Options<'_>) -> usize {
    let content = line.to_string();
    if content.is_empty() {
        return 1;
    }
    let wrapped = textwrap::wrap(&content, options);
    wrapped.len().max(1)
}
