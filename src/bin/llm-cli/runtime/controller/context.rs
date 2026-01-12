use crate::runtime::context::SUMMARY_SAMPLE_COUNT;
use crate::runtime::{
    compact_conversation, context_limit, summarize_conversation_head, usage_for, OverlayState,
    PagerState,
};

use super::AppController;

impl AppController {
    pub fn show_context_status(&mut self) -> bool {
        let Some(conv) = self.state.active_conversation() else {
            return false;
        };
        let usage = usage_for(conv, &self.state.config);
        let threshold = self
            .state
            .config
            .chat
            .auto_compact_threshold
            .clamp(0.0, 1.0)
            * 100.0;
        let text = format!(
            "Provider: {}\nModel: {}\nUsed: {} tokens\nLimit: {} tokens\nUsage: {:.1}%\nTrim: {:?}\nAuto-compact: {:.0}%",
            conv.provider_id,
            conv.model.clone().unwrap_or_else(|| "default".to_string()),
            usage.used_tokens,
            usage.max_tokens,
            usage.percent(),
            self.state.config.chat.trim_strategy,
            threshold
        );
        self.state.overlay = OverlayState::Pager(PagerState::new("Context status", &text));
        true
    }

    pub fn summarize_context(&mut self, count: Option<usize>) -> bool {
        let Some(conv) = self.state.active_conversation_mut() else {
            return false;
        };
        let count = count.unwrap_or(SUMMARY_SAMPLE_COUNT);
        if !summarize_conversation_head(conv, count) {
            return false;
        }
        self.record_snapshot();
        self.push_notice(format!("Summarized first {count} messages."));
        self.state.scroll.reset();
        true
    }

    pub fn compact_context(&mut self) -> bool {
        let max_tokens = {
            let conv = match self.state.active_conversation() {
                Some(conv) => conv,
                None => return false,
            };
            context_limit(conv, &self.state.config)
        };
        let Some(conv) = self.state.active_conversation_mut() else {
            return false;
        };
        if !compact_conversation(conv, max_tokens) {
            return false;
        }
        self.record_snapshot();
        self.push_notice("Context compacted.".to_string());
        self.state.scroll.reset();
        true
    }

    pub fn maybe_auto_compact(&mut self) -> bool {
        let threshold = self
            .state
            .config
            .chat
            .auto_compact_threshold
            .clamp(0.0, 1.0);
        if threshold <= 0.0 {
            return false;
        }
        let usage = {
            let conv = match self.state.active_conversation() {
                Some(conv) => conv,
                None => return false,
            };
            usage_for(conv, &self.state.config)
        };
        if usage.percent() < threshold * 100.0 {
            return false;
        }
        let Some(conv) = self.state.active_conversation_mut() else {
            return false;
        };
        if compact_conversation(conv, usage.max_tokens) {
            self.record_snapshot();
            self.push_notice("Context auto-compacted.".to_string());
            self.state.scroll.reset();
            return true;
        }
        false
    }
}
