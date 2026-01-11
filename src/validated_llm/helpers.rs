use crate::error::LLMError;

pub(super) fn decrement_attempts(remaining: usize, err: &str) -> Result<usize, LLMError> {
    let remaining = remaining.saturating_sub(1);
    if remaining == 0 {
        return Err(LLMError::InvalidRequest(format!(
            "Validation error after max attempts: {err}"
        )));
    }
    Ok(remaining)
}
