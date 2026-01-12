mod loader;
mod types;

use std::path::Path;

pub use loader::{load_skills, SkillError};
pub use types::Skill;

#[derive(Debug, Default, Clone)]
pub struct SkillCatalog {
    skills: Vec<Skill>,
}

impl SkillCatalog {
    pub fn load(dir: &Path) -> Result<Self, SkillError> {
        let skills = load_skills(dir)?;
        Ok(Self { skills })
    }

    pub fn list(&self) -> &[Skill] {
        &self.skills
    }

    pub fn find(&self, name: &str) -> Option<&Skill> {
        self.skills.iter().find(|skill| skill.matches(name))
    }
}
