use std::borrow::Borrow;
use std::collections::HashMap;

use crate::{error::LLMError, LLMProvider};

#[cfg(feature = "api")]
use crate::api::Server;

/// Stores multiple LLM backends (OpenAI, Anthropic, etc.) identified by a key.
#[derive(Default)]
pub struct LLMRegistry {
    pub backends: HashMap<ProviderId, Box<dyn LLMProvider>>,
}

impl LLMRegistry {
    pub fn new() -> Self {
        Self {
            backends: HashMap::new(),
        }
    }

    /// Inserts a backend under an identifier, e.g. "openai".
    pub fn insert(&mut self, id: impl Into<String>, llm: Box<dyn LLMProvider>) {
        if let Err(err) = self.try_insert(id, llm) {
            log::warn!("Invalid provider id: {err}");
        }
    }

    /// Inserts a backend and returns an error for invalid ids.
    pub fn try_insert(
        &mut self,
        id: impl Into<String>,
        llm: Box<dyn LLMProvider>,
    ) -> Result<(), LLMError> {
        let id = ProviderId::new(id)?;
        self.backends.insert(id, llm);
        Ok(())
    }

    /// Retrieves a backend by its identifier.
    pub fn get(&self, id: &str) -> Option<&dyn LLMProvider> {
        self.backends.get(id).map(|b| b.as_ref())
    }

    #[cfg(feature = "api")]
    /// Starts a REST API server on the specified address.
    pub async fn serve(self, addr: impl Into<String>) -> Result<(), LLMError> {
        let server = Server::new(self);
        server.run(&addr.into()).await?;
        Ok(())
    }
}

/// Builder pattern for LLMRegistry.
#[derive(Default)]
pub struct LLMRegistryBuilder {
    registry: LLMRegistry,
}

impl LLMRegistryBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a backend under the given id.
    pub fn register(mut self, id: impl Into<String>, llm: Box<dyn LLMProvider>) -> Self {
        self.registry.insert(id, llm);
        self
    }

    /// Adds a backend with validation.
    pub fn try_register(
        mut self,
        id: impl Into<String>,
        llm: Box<dyn LLMProvider>,
    ) -> Result<Self, LLMError> {
        self.registry.try_insert(id, llm)?;
        Ok(self)
    }

    /// Builds the final LLMRegistry.
    pub fn build(self) -> LLMRegistry {
        self.registry
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProviderId(String);

impl ProviderId {
    fn new(id: impl Into<String>) -> Result<Self, LLMError> {
        let id = id.into();
        if id.trim().is_empty() {
            return Err(LLMError::InvalidRequest(
                "Provider id cannot be empty".to_string(),
            ));
        }
        Ok(Self(id))
    }
}

impl Borrow<str> for ProviderId {
    fn borrow(&self) -> &str {
        &self.0
    }
}
