// src/backends/bedrock/error.rs
//! Error types for the AWS Bedrock backend

use std::fmt;

/// Result type alias for Bedrock operations
pub type Result<T> = std::result::Result<T, BedrockError>;

/// Errors that can occur when using the Bedrock backend
#[derive(Debug)]
pub enum BedrockError {
    /// AWS configuration error
    ConfigurationError(String),

    /// Invalid request parameters
    InvalidRequest(String),

    /// Invalid response from Bedrock
    InvalidResponse(String),

    /// API error from AWS
    ApiError(String),

    /// Unsupported operation for the model
    UnsupportedOperation(String),

    /// Streaming error
    StreamError(String),

    /// JSON serialization/deserialization error
    SerdeError(serde_json::Error),
}

impl fmt::Display for BedrockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            Self::InvalidRequest(msg) => write!(f, "Invalid request: {}", msg),
            Self::InvalidResponse(msg) => write!(f, "Invalid response: {}", msg),
            Self::ApiError(msg) => write!(f, "API error: {}", msg),
            Self::UnsupportedOperation(msg) => write!(f, "Unsupported operation: {}", msg),
            Self::StreamError(msg) => write!(f, "Stream error: {}", msg),
            Self::SerdeError(e) => write!(f, "Serialization error: {}", e),
        }
    }
}

impl std::error::Error for BedrockError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::SerdeError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<serde_json::Error> for BedrockError {
    fn from(error: serde_json::Error) -> Self {
        Self::SerdeError(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = BedrockError::InvalidRequest("test error".to_string());
        assert_eq!(err.to_string(), "Invalid request: test error");
    }

    #[test]
    fn test_serde_error_conversion() {
        let json_err = serde_json::from_str::<serde_json::Value>("invalid json");
        assert!(json_err.is_err());

        let bedrock_err: BedrockError = json_err.unwrap_err().into();
        assert!(matches!(bedrock_err, BedrockError::SerdeError(_)));
    }
}
