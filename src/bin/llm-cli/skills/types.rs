use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(default)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub prompt: String,
    pub aliases: Vec<String>,
}

impl Skill {
    pub fn matches(&self, input: &str) -> bool {
        let needle = normalize(input);
        if normalize(&self.name) == needle {
            return true;
        }
        self.aliases.iter().any(|alias| normalize(alias) == needle)
    }
}

fn normalize(value: &str) -> String {
    value.trim().to_lowercase()
}
