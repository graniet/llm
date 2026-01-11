use super::llm_builder::LLMBuilder;

impl LLMBuilder {
    /// Sets the encoding format for embedding outputs.
    pub fn embedding_encoding_format(mut self, format: impl Into<String>) -> Self {
        self.state.embedding_encoding_format = Some(format.into());
        self
    }

    /// Sets the dimensions for embedding outputs.
    pub fn embedding_dimensions(mut self, embedding_dimensions: u32) -> Self {
        self.state.embedding_dimensions = Some(embedding_dimensions);
        self
    }
}
