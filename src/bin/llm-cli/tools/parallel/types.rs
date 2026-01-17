//! Types for parallel tool execution.

/// Configuration for parallel execution.
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Maximum number of concurrent non-mutating operations.
    pub max_concurrent_reads: usize,
    /// Maximum number of concurrent mutating operations (usually 1).
    pub max_concurrent_writes: usize,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            max_concurrent_reads: 8,
            max_concurrent_writes: 1,
        }
    }
}

/// Tool invocation for parallel execution.
#[derive(Debug, Clone)]
pub struct ToolInvocation {
    /// Unique identifier.
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Arguments as JSON string.
    pub arguments: String,
}

/// Result of a single tool execution.
#[derive(Debug, Clone)]
pub struct ToolExecutionResult {
    /// Invocation ID.
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Result output or error.
    pub result: Result<String, String>,
}
