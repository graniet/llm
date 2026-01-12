use std::fs;
use std::path::Path;

use thiserror::Error;

use super::types::Skill;

#[derive(Debug, Error)]
pub enum SkillError {
    #[error("skills io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("skills parse error: {0}")]
    Parse(#[from] serde_yaml::Error),
}

pub fn load_skills(dir: &Path) -> Result<Vec<Skill>, SkillError> {
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut skills = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let ext = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
        if ext != "yaml" && ext != "yml" {
            continue;
        }
        let contents = fs::read_to_string(&path)?;
        let mut skill: Skill = serde_yaml::from_str(&contents)?;
        if skill.name.trim().is_empty() {
            if let Some(stem) = path.file_stem().and_then(|v| v.to_str()) {
                skill.name = stem.to_string();
            }
        }
        skills.push(skill);
    }
    Ok(skills)
}
