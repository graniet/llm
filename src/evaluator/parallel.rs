#[path = "parallel/types.rs"]
mod types;

#[path = "parallel/evaluator.rs"]
mod evaluator;

pub use evaluator::ParallelEvaluator;
pub use types::ParallelEvalResult;
