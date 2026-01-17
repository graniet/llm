//! Parallel tool execution with mutation awareness.
//!
//! Provides intelligent parallel execution of tools:
//! - Non-mutating tools (read-only) can run in parallel
//! - Mutating tools (write) run sequentially to prevent conflicts

#![allow(dead_code)]

mod mutation;
pub mod types;

use std::collections::HashSet;
use std::sync::Arc;

use tokio::sync::Semaphore;

use super::context::ToolContext;
use super::registry::ToolRegistry;

pub use mutation::is_mutating_tool;
pub use types::{ParallelConfig, ToolExecutionResult, ToolInvocation};

/// Parallel tool executor with mutation-aware scheduling.
pub struct ParallelExecutor {
    /// Configuration.
    config: ParallelConfig,
    /// Semaphore for read operations.
    read_semaphore: Arc<Semaphore>,
    /// Semaphore for write operations.
    write_semaphore: Arc<Semaphore>,
}

impl ParallelExecutor {
    /// Create a new parallel executor with default configuration.
    pub fn new() -> Self {
        Self::with_config(ParallelConfig::default())
    }

    /// Create a new parallel executor with custom configuration.
    pub fn with_config(config: ParallelConfig) -> Self {
        Self {
            read_semaphore: Arc::new(Semaphore::new(config.max_concurrent_reads)),
            write_semaphore: Arc::new(Semaphore::new(config.max_concurrent_writes)),
            config,
        }
    }

    /// Execute multiple tools with parallel/sequential awareness.
    pub async fn execute_batch(
        &self,
        invocations: Vec<ToolInvocation>,
        registry: &ToolRegistry,
        context: &ToolContext,
    ) -> Vec<ToolExecutionResult> {
        let mut handles = Vec::with_capacity(invocations.len());
        let mut results = Vec::with_capacity(invocations.len());
        let result_order: Vec<String> = invocations.iter().map(|i| i.id.clone()).collect();

        // Group by mutating vs non-mutating
        let (non_mutating, mutating): (Vec<_>, Vec<_>) = invocations
            .into_iter()
            .partition(|inv| !is_mutating_tool(&inv.name));

        // Execute non-mutating tools in parallel
        for invocation in non_mutating {
            let registry = registry.clone();
            let context = context.clone();
            let semaphore = Arc::clone(&self.read_semaphore);

            let handle = tokio::spawn(async move {
                let _permit = semaphore.acquire().await;
                execute_single(invocation, &registry, &context)
            });
            handles.push(handle);
        }

        // Execute mutating tools sequentially
        for invocation in mutating {
            let registry = registry.clone();
            let context = context.clone();
            let semaphore = Arc::clone(&self.write_semaphore);

            let handle = tokio::spawn(async move {
                let _permit = semaphore.acquire().await;
                execute_single(invocation, &registry, &context)
            });
            handles.push(handle);
        }

        // Collect results
        for handle in handles {
            if let Ok(result) = handle.await {
                results.push(result);
            }
        }

        // Reorder results to match input order
        let mut ordered_results = Vec::with_capacity(results.len());
        for id in result_order {
            if let Some(result) = results.iter().find(|r| r.id == id) {
                ordered_results.push(result.clone());
            }
        }

        ordered_results
    }

    /// Execute tools with dependencies (topological order).
    pub async fn execute_with_deps(
        &self,
        invocations: Vec<ToolInvocation>,
        dependencies: &[(String, Vec<String>)],
        registry: &ToolRegistry,
        context: &ToolContext,
    ) -> Vec<ToolExecutionResult> {
        let mut completed: HashSet<String> = HashSet::new();
        let mut results = Vec::new();
        let mut pending: Vec<_> = invocations.into_iter().collect();
        let dep_map: std::collections::HashMap<_, _> = dependencies.iter().cloned().collect();

        while !pending.is_empty() {
            // Find tools that can run now (all dependencies satisfied)
            let (ready, not_ready): (Vec<_>, Vec<_>) = pending.into_iter().partition(|inv| {
                dep_map
                    .get(&inv.id)
                    .map(|deps| deps.iter().all(|d| completed.contains(d)))
                    .unwrap_or(true)
            });

            if ready.is_empty() && !not_ready.is_empty() {
                // Circular dependency or unsatisfied deps - execute remaining anyway
                pending = not_ready;
                for inv in std::mem::take(&mut pending) {
                    let result = execute_single(inv, registry, context);
                    completed.insert(result.id.clone());
                    results.push(result);
                }
                break;
            }

            // Execute ready tools
            let batch_results = self.execute_batch(ready, registry, context).await;
            for result in batch_results {
                completed.insert(result.id.clone());
                results.push(result);
            }

            pending = not_ready;
        }

        results
    }
}

impl Default for ParallelExecutor {
    fn default() -> Self {
        Self::new()
    }
}

fn execute_single(
    invocation: ToolInvocation,
    registry: &ToolRegistry,
    context: &ToolContext,
) -> ToolExecutionResult {
    let result = registry.execute(&invocation.name, &invocation.arguments, context);
    ToolExecutionResult {
        id: invocation.id,
        name: invocation.name,
        result: result.map_err(|e| e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_config_default() {
        let config = ParallelConfig::default();
        assert_eq!(config.max_concurrent_reads, 8);
        assert_eq!(config.max_concurrent_writes, 1);
    }
}
