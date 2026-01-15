use fuzzy_matcher::skim::SkimMatcherV2;
use fuzzy_matcher::FuzzyMatcher;

#[derive(Debug, Clone)]
pub struct PickerItem {
    pub id: String,
    pub label: String,
    pub meta: Option<String>,
    pub badges: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct PickerState {
    pub title: String,
    pub query: String,
    pub items: Vec<PickerItem>,
    pub filtered: Vec<PickerItem>,
    pub selected: usize,
}

impl PickerState {
    pub fn new(title: impl Into<String>, items: Vec<PickerItem>) -> Self {
        let mut state = Self {
            title: title.into(),
            query: String::new(),
            filtered: items.clone(),
            items,
            selected: 0,
        };
        state.refresh();
        state
    }

    pub fn push_query(&mut self, ch: char) {
        self.query.push(ch);
        self.refresh();
    }

    pub fn pop_query(&mut self) {
        self.query.pop();
        self.refresh();
    }

    pub fn next(&mut self) {
        if !self.filtered.is_empty() {
            self.selected = (self.selected + 1).min(self.filtered.len() - 1);
        }
    }

    pub fn prev(&mut self) {
        if self.selected > 0 {
            self.selected -= 1;
        }
    }

    pub fn selected_item(&self) -> Option<&PickerItem> {
        self.filtered.get(self.selected)
    }

    fn refresh(&mut self) {
        if self.query.is_empty() {
            self.filtered = self.items.clone();
            return;
        }
        let matcher = SkimMatcherV2::default();
        let mut scored: Vec<(i64, PickerItem)> = self
            .items
            .iter()
            .filter_map(|item| {
                matcher
                    .fuzzy_match(&item.label, &self.query)
                    .map(|score| (score, item.clone()))
            })
            .collect();
        scored.sort_by(|a, b| b.0.cmp(&a.0));
        self.filtered = scored.into_iter().map(|(_, item)| item).collect();
        self.selected = self.selected.min(self.filtered.len().saturating_sub(1));
    }
}
