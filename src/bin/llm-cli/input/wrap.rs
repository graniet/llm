use std::ops::Range;

use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

pub fn wrap_ranges(text: &str, width: u16) -> Vec<Range<usize>> {
    let max_width = width.max(1) as usize;
    let mut ranges = Vec::new();
    let mut line_start = 0;
    let mut line_width = 0usize;

    for (idx, grapheme) in text.grapheme_indices(true) {
        if grapheme == "\n" {
            ranges.push(line_start..idx);
            line_start = idx + grapheme.len();
            line_width = 0;
            continue;
        }
        let g_width = grapheme.width();
        if line_width + g_width > max_width && line_start < idx {
            ranges.push(line_start..idx);
            line_start = idx;
            line_width = 0;
        }
        line_width = line_width.saturating_add(g_width);
    }

    if line_start <= text.len() {
        ranges.push(line_start..text.len());
    }
    if ranges.is_empty() {
        ranges.push(0..0);
    }
    ranges
}
