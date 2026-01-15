#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "ollama")]
pub mod ollama;

#[cfg(feature = "deepseek")]
pub mod deepseek;

#[cfg(feature = "xai")]
pub mod xai;

#[cfg(feature = "phind")]
pub mod phind;

#[cfg(feature = "google")]
pub mod google;

#[cfg(feature = "groq")]
pub mod groq;

#[cfg(feature = "openrouter")]
pub mod openrouter;

#[cfg(feature = "azure_openai")]
pub mod azure_openai;

#[cfg(feature = "elevenlabs")]
pub mod elevenlabs;

#[cfg(feature = "cohere")]
pub mod cohere;

#[cfg(feature = "mistral")]
pub mod mistral;

#[cfg(feature = "huggingface")]
pub mod huggingface;

#[cfg(feature = "bedrock")]
pub mod aws;
