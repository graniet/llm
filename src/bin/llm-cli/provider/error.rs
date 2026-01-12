#[derive(Debug, thiserror::Error)]
pub enum ProviderBuildError {
    #[error("unknown provider: {0}")]
    UnknownProvider(String),
    #[error("missing API key for provider: {0}")]
    MissingApiKey(String),
    #[error("provider build error: {0}")]
    Build(String),
}
