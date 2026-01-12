use crate::skills::Skill;

use super::AppController;

impl AppController {
    pub fn activate_skill(&mut self, skill: &Skill) -> bool {
        let conv_id = match self.state.active_conversation_mut() {
            Some(conv) => {
                let prompt = compose_prompt(conv.system_prompt.as_deref(), &skill.prompt);
                conv.system_prompt = Some(prompt);
                conv.id
            }
            None => return false,
        };
        self.state.provider_cache.remove(&conv_id);
        true
    }

    pub fn find_skill(&self, name: &str) -> Option<&Skill> {
        self.state.skills.find(name)
    }

    pub fn extract_skill_mention(&self, text: &str) -> Option<(&Skill, String)> {
        let mut cleaned = Vec::new();
        let mut matched = None;
        for token in text.split_whitespace() {
            if let Some(skill_name) = token.strip_prefix('@') {
                if let Some(skill) = self.find_skill(skill_name) {
                    matched = Some(skill);
                    continue;
                }
            }
            cleaned.push(token);
        }
        matched.map(|skill| (skill, cleaned.join(" ")))
    }
}

fn compose_prompt(base: Option<&str>, skill_prompt: &str) -> String {
    match base {
        Some(base) if !base.trim().is_empty() => format!("{base}\n\n{skill_prompt}"),
        _ => skill_prompt.to_string(),
    }
}
