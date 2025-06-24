use crate::chat::{ChatMessage, ChatProvider, ChatResponse, Tool};
use crate::completion::{CompletionProvider, CompletionRequest, CompletionResponse};
use crate::embedding::EmbeddingProvider;
#[cfg(feature = "elevenlabs")]
use crate::error::LLMError;
use crate::health::HealthProvider;
use crate::models::ModelsProvider;
use crate::stt::SpeechToTextProvider;
use crate::tts::TextToSpeechProvider;
use crate::LLMProvider;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// ElevenLabs speech to text backend implementation
///
/// This struct provides functionality for speech-to-text transcription using the ElevenLabs API.
/// It implements various LLM provider traits but only supports speech-to-text functionality.
pub struct ElevenLabs {
    /// API key for ElevenLabs authentication
    api_key: String,
    /// Model identifier for speech-to-text
    model_id: String,
    /// Base URL for API requests
    base_url: String,
    /// Optional timeout duration in seconds
    timeout_seconds: Option<u64>,
    /// HTTP client for making requests
    client: Client,
    /// Voice ID to use for speech synthesis
    voice: Option<String>,
}

/// Internal representation of a word from ElevenLabs API response
#[derive(Debug, Deserialize)]
struct ElevenLabsWord {
    /// The transcribed word text
    text: String,
    /// Start time of the word in seconds
    #[serde(default)]
    start: f32,
    /// End time of the word in seconds
    #[serde(default)]
    end: f32,
}

/// Public representation of a transcribed word with timing information
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Word {
    /// The transcribed word text
    pub text: String,
    /// Start time of the word in seconds
    pub start: f32,
    /// End time of the word in seconds
    pub end: f32,
}

/// Response structure from ElevenLabs speech-to-text API
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct ElevenLabsResponse {
    /// Detected language code if available
    #[serde(skip_serializing_if = "Option::is_none")]
    language_code: Option<String>,
    /// Probability of the detected language if available
    #[serde(skip_serializing_if = "Option::is_none")]
    language_probability: Option<f32>,
    /// Full transcribed text
    #[serde(skip_serializing_if = "Option::is_none")]
    text: String,
    /// Optional list of words with timing information
    words: Option<Vec<ElevenLabsWord>>,
}

impl ElevenLabs {
    /// Creates a new ElevenLabs instance
    ///
    /// # Arguments
    ///
    /// * `api_key` - API key for ElevenLabs authentication
    /// * `model_id` - Model identifier for speech-to-text
    /// * `base_url` - Base URL for API requests
    /// * `timeout_seconds` - Optional timeout duration in seconds
    ///
    /// # Returns
    ///
    /// A new ElevenLabs instance
    pub fn new(
        api_key: String,
        model_id: String,
        base_url: String,
        timeout_seconds: Option<u64>,
        voice: Option<String>,
    ) -> Self {
        Self {
            api_key,
            model_id,
            base_url,
            timeout_seconds,
            client: Client::new(),
            voice,
        }
    }
}

#[async_trait]
impl SpeechToTextProvider for ElevenLabs {
    /// Transcribes audio data to text using ElevenLabs API
    ///
    /// # Arguments
    ///
    /// * `audio` - Raw audio data as bytes
    ///
    /// # Returns
    ///
    /// * `Ok(String)` - Transcribed text
    /// * `Err(LLMError)` - Error if transcription fails
    async fn transcribe(&self, audio: Vec<u8>) -> Result<String, LLMError> {
        let url = format!("{}/speech-to-text", self.base_url);
        let part = reqwest::multipart::Part::bytes(audio).file_name("audio.wav");
        let form = reqwest::multipart::Form::new()
            .text("model_id", self.model_id.clone())
            .part("file", part);

        let mut req = self
            .client
            .post(url)
            .header("xi-api-key", &self.api_key)
            .multipart(form);

        if let Some(t) = self.timeout_seconds {
            req = req.timeout(Duration::from_secs(t));
        }

        let resp = req.send().await?.error_for_status()?;
        let text = resp.text().await?;
        let raw = text.clone();
        let parsed: ElevenLabsResponse =
            serde_json::from_str(&text).map_err(|e| LLMError::ResponseFormatError {
                message: e.to_string(),
                raw_response: raw,
            })?;

        let words: Option<Vec<Word>> = parsed.words.map(|ws| {
            ws.into_iter()
                .map(|w| Word {
                    text: w.text,
                    start: w.start,
                    end: w.end,
                })
                .collect()
        });

        Ok(words
            .unwrap_or_default()
            .into_iter()
            .map(|w| w.text)
            .collect())
    }

    /// Transcribes audio file to text using ElevenLabs API
    ///
    /// # Arguments
    ///
    /// * `file_path` - Path to the audio file
    ///
    /// # Returns
    ///
    /// * `Ok(String)` - Transcribed text
    /// * `Err(LLMError)` - Error if transcription fails
    async fn transcribe_file(&self, file_path: &str) -> Result<String, LLMError> {
        let url = format!("{}/speech-to-text", self.base_url);
        let form = reqwest::multipart::Form::new()
            .text("model_id", self.model_id.clone())
            .file("file", file_path)
            .await
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let mut req = self
            .client
            .post(url)
            .header("xi-api-key", &self.api_key)
            .multipart(form);

        if let Some(t) = self.timeout_seconds {
            req = req.timeout(Duration::from_secs(t));
        }

        let resp = req.send().await?.error_for_status()?;
        let text = resp.text().await?;
        let raw = text.clone();
        let parsed: ElevenLabsResponse =
            serde_json::from_str(&text).map_err(|e| LLMError::ResponseFormatError {
                message: e.to_string(),
                raw_response: raw,
            })?;

        let words: Option<Vec<Word>> = parsed.words.map(|ws| {
            ws.into_iter()
                .map(|w| Word {
                    text: w.text,
                    start: w.start,
                    end: w.end,
                })
                .collect()
        });

        Ok(words
            .unwrap_or_default()
            .into_iter()
            .map(|w| w.text)
            .collect())
    }
}

#[async_trait]
impl CompletionProvider for ElevenLabs {
    /// Returns a not implemented message for completion requests
    async fn complete(&self, _req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        Ok(CompletionResponse {
            text: "ElevenLabs completion not implemented.".into(),
        })
    }
}

#[async_trait]
impl EmbeddingProvider for ElevenLabs {
    /// Returns an error indicating embedding is not supported
    async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Err(LLMError::ProviderError(
            "Embedding not supported".to_string(),
        ))
    }
}

#[async_trait]
impl ChatProvider for ElevenLabs {
    /// Returns an error indicating chat is not supported
    async fn chat(&self, _messages: &[ChatMessage]) -> Result<Box<dyn ChatResponse>, LLMError> {
        Err(LLMError::ProviderError("Chat not supported".to_string()))
    }

    /// Returns an error indicating chat with tools is not supported
    async fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        Err(LLMError::ProviderError(
            "Chat with tools not supported".to_string(),
        ))
    }
}

#[async_trait]
impl ModelsProvider for ElevenLabs {}

#[async_trait]
impl HealthProvider for ElevenLabs {}

impl LLMProvider for ElevenLabs {
    /// Returns None as no tools are supported
    fn tools(&self) -> Option<&[Tool]> {
        None
    }
}

#[async_trait]
impl TextToSpeechProvider for ElevenLabs {
    /// Converts text to speech using ElevenLabs API
    ///
    /// # Arguments
    ///
    /// * `text` - Text to convert to speech
    /// * `voice_id` - Voice ID to use for speech synthesis
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<u8>)` - Audio data as bytes
    /// * `Err(LLMError)` - Error if conversion fails
    async fn speech(&self, text: &str) -> Result<Vec<u8>, LLMError> {
        let url = format!(
            "{}/text-to-speech/{}?output_format=mp3_44100_128",
            self.base_url,
            self.voice
                .clone()
                .unwrap_or("JBFqnCBsd6RMkjVDRZzb".to_string())
        );

        let body = serde_json::json!({
            "text": text,
            "model_id": self.model_id
        });

        let mut req = self
            .client
            .post(url)
            .header("xi-api-key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&body);

        if let Some(t) = self.timeout_seconds {
            req = req.timeout(Duration::from_secs(t));
        }

        let resp = req.send().await?.error_for_status()?;
        let audio_data = resp.bytes().await?;

        Ok(audio_data.to_vec())
    }
}
