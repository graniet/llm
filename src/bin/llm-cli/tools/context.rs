#[derive(Debug, Clone)]
pub struct ToolContext {
    pub allowed_paths: Vec<String>,
    pub timeout_ms: u64,
    pub working_dir: String,
}
