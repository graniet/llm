/// Result of a parallel evaluation including response, score, and timing information.
#[derive(Debug)]
pub struct ParallelEvalResult {
    /// The text response from the LLM.
    pub text: String,
    /// Score assigned by the scoring function.
    pub score: f32,
    /// Time taken to generate the response in milliseconds.
    pub time_ms: u128,
    /// Identifier of the provider that generated this response.
    pub provider_id: String,
}
