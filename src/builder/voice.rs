use super::llm_builder::LLMBuilder;

impl LLMBuilder {
    /// Set the voice for TTS providers.
    pub fn voice(mut self, voice: impl Into<String>) -> Self {
        self.state.voice = Some(voice.into());
        self
    }
}
