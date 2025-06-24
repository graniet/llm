use crate::error::LLMError;
#[async_trait::async_trait]
pub trait HealthProvider {
    async fn health_check(&self) -> Result<(), LLMError> {
        Err(LLMError::Generic(
            "health check not implemented for this provider".to_string(),
        ))
    }
}
