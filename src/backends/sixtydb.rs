use std::sync::Arc;

use crate::chat::{ChatMessage, ChatProvider, ChatResponse, Tool};
use crate::completion::{CompletionProvider, CompletionRequest, CompletionResponse};
use crate::embedding::EmbeddingProvider;
#[cfg(feature = "sixtydb")]
use crate::error::LLMError;
use crate::models::ModelsProvider;
use crate::stt::SpeechToTextProvider;
use crate::tts::TextToSpeechProvider;
use crate::LLMProvider;
use async_trait::async_trait;
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use reqwest::Client;
use serde::Deserialize;
use std::time::Duration;

/// Configuration for the 60db client.
#[derive(Debug)]
pub struct SixtyDbConfig {
    /// API key for authentication (sent as a Bearer token).
    pub api_key: String,
    /// Model identifier (kept for builder parity; 60db's TTS endpoint
    /// selects the model implicitly from the voice, so it is not sent).
    pub model_id: String,
    /// Base URL for API requests.
    pub base_url: String,
    /// Request timeout in seconds.
    pub timeout_seconds: Option<u64>,
    /// Voice setting for TTS (maps to `voice_id`).
    pub voice: Option<String>,
}

/// 60db audio backend implementation
///
/// This struct provides text-to-speech synthesis and speech-to-text
/// transcription using the [60db](https://60db.ai) API. It implements every
/// LLM provider trait for uniformity but only supports audio (TTS + STT); all
/// other capabilities (chat, completion, embedding) return an unsupported error.
///
/// The client uses `Arc` internally for configuration, making cloning cheap.
#[derive(Debug, Clone)]
pub struct SixtyDb {
    /// Shared configuration wrapped in Arc for cheap cloning.
    pub config: Arc<SixtyDbConfig>,
    /// HTTP client for making requests.
    pub client: Client,
}

/// Response structure from the 60db text-to-speech API.
///
/// The endpoint returns the synthesized audio as a base64-encoded payload
/// rather than raw bytes, so it must be decoded before use.
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct SixtyDbTtsResponse {
    /// Whether the request succeeded.
    #[serde(default)]
    success: bool,
    /// Human-readable status description.
    #[serde(default)]
    message: Option<String>,
    /// Base64-encoded audio payload.
    #[serde(default)]
    audio_base64: Option<String>,
    /// Sample rate of the returned audio in Hz.
    #[serde(default)]
    sample_rate: Option<u32>,
    /// Length of the returned audio in seconds.
    #[serde(default)]
    duration_seconds: Option<f32>,
    /// Encoding of the returned audio.
    #[serde(default)]
    encoding: Option<String>,
    /// Format of the returned audio (mp3, wav, ogg, flac).
    #[serde(default)]
    output_format: Option<String>,
}

/// Response structure from the 60db speech-to-text API.
///
/// The endpoint returns the full transcript directly in the `text` field
/// (alongside richer per-segment / per-word data we currently ignore).
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct SixtyDbSttResponse {
    /// Unique identifier for the request.
    #[serde(default)]
    request_id: Option<String>,
    /// Full transcribed text.
    text: String,
    /// Detected language code if available.
    #[serde(default)]
    language: Option<String>,
    /// Human-readable detected language name if available.
    #[serde(default)]
    language_name: Option<String>,
    /// Audio duration in seconds.
    #[serde(default)]
    duration_sec: Option<f32>,
}

impl SixtyDb {
    /// Creates a new 60db instance.
    ///
    /// # Arguments
    ///
    /// * `api_key` - API key for 60db authentication
    /// * `model_id` - Model identifier (kept for builder parity)
    /// * `base_url` - Base URL for API requests
    /// * `timeout_seconds` - Optional timeout duration in seconds
    /// * `voice` - Optional voice id used for synthesis
    ///
    /// # Returns
    ///
    /// A new 60db instance
    pub fn new(
        api_key: String,
        model_id: String,
        base_url: String,
        timeout_seconds: Option<u64>,
        voice: Option<String>,
    ) -> Self {
        Self::with_client(
            Client::new(),
            api_key,
            model_id,
            base_url,
            timeout_seconds,
            voice,
        )
    }

    /// Creates a new 60db instance with a custom HTTP client.
    pub fn with_client(
        client: Client,
        api_key: String,
        model_id: String,
        base_url: String,
        timeout_seconds: Option<u64>,
        voice: Option<String>,
    ) -> Self {
        Self {
            config: Arc::new(SixtyDbConfig {
                api_key,
                model_id,
                base_url,
                timeout_seconds,
                voice,
            }),
            client,
        }
    }

    pub fn api_key(&self) -> &str {
        &self.config.api_key
    }

    pub fn model_id(&self) -> &str {
        &self.config.model_id
    }

    pub fn base_url(&self) -> &str {
        &self.config.base_url
    }

    pub fn timeout_seconds(&self) -> Option<u64> {
        self.config.timeout_seconds
    }

    pub fn voice(&self) -> Option<&str> {
        self.config.voice.as_deref()
    }

    pub fn client(&self) -> &Client {
        &self.client
    }
}

#[async_trait]
impl SpeechToTextProvider for SixtyDb {
    /// Transcribes audio data to text using the 60db API.
    ///
    /// Uploads the raw audio bytes to the `/stt` endpoint as multipart form
    /// data and returns the full transcript from the response.
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
        let url = format!("{}/stt", self.config.base_url);
        let part = reqwest::multipart::Part::bytes(audio).file_name("audio.wav");
        let form = reqwest::multipart::Form::new().part("file", part);

        let mut req = self
            .client
            .post(url)
            .bearer_auth(&self.config.api_key)
            .multipart(form);

        if let Some(t) = self.config.timeout_seconds {
            req = req.timeout(Duration::from_secs(t));
        }

        let resp = req.send().await?.error_for_status()?;
        let body = resp.text().await?;
        let parsed: SixtyDbSttResponse =
            serde_json::from_str(&body).map_err(|e| LLMError::ResponseFormatError {
                message: e.to_string(),
                raw_response: body,
            })?;

        Ok(parsed.text)
    }

    /// Transcribes an audio file to text using the 60db API.
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
        let url = format!("{}/stt", self.config.base_url);
        let form = reqwest::multipart::Form::new()
            .file("file", file_path)
            .await
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let mut req = self
            .client
            .post(url)
            .bearer_auth(&self.config.api_key)
            .multipart(form);

        if let Some(t) = self.config.timeout_seconds {
            req = req.timeout(Duration::from_secs(t));
        }

        let resp = req.send().await?.error_for_status()?;
        let body = resp.text().await?;
        let parsed: SixtyDbSttResponse =
            serde_json::from_str(&body).map_err(|e| LLMError::ResponseFormatError {
                message: e.to_string(),
                raw_response: body,
            })?;

        Ok(parsed.text)
    }
}

#[async_trait]
impl CompletionProvider for SixtyDb {
    /// Returns a not implemented message for completion requests
    async fn complete(&self, _req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        Ok(CompletionResponse {
            text: "60db completion not implemented.".into(),
        })
    }
}

#[async_trait]
impl EmbeddingProvider for SixtyDb {
    /// Returns an error indicating embedding is not supported
    async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Err(LLMError::ProviderError(
            "Embedding not supported".to_string(),
        ))
    }
}

#[async_trait]
impl ChatProvider for SixtyDb {
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
impl ModelsProvider for SixtyDb {}

impl LLMProvider for SixtyDb {
    /// Returns None as no tools are supported
    fn tools(&self) -> Option<&[Tool]> {
        None
    }
}

#[async_trait]
impl TextToSpeechProvider for SixtyDb {
    /// Converts text to speech using the 60db API.
    ///
    /// Sends the text to the `/tts-synthesize` endpoint, then decodes the
    /// base64-encoded audio payload from the JSON response.
    ///
    /// # Arguments
    ///
    /// * `text` - Text to convert to speech
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<u8>)` - Decoded audio data as bytes
    /// * `Err(LLMError)` - Error if conversion fails
    async fn speech(&self, text: &str) -> Result<Vec<u8>, LLMError> {
        let url = format!("{}/tts-synthesize", self.config.base_url);

        let mut body = serde_json::json!({
            "text": text,
            "output_format": "mp3",
        });
        if let Some(voice) = &self.config.voice {
            body["voice_id"] = serde_json::Value::String(voice.clone());
        }

        let mut req = self
            .client
            .post(url)
            .bearer_auth(&self.config.api_key)
            .header("Content-Type", "application/json")
            .json(&body);

        if let Some(t) = self.config.timeout_seconds {
            req = req.timeout(Duration::from_secs(t));
        }

        let resp = req.send().await?.error_for_status()?;
        let text_body = resp.text().await?;
        let raw = text_body.clone();
        let parsed: SixtyDbTtsResponse =
            serde_json::from_str(&text_body).map_err(|e| LLMError::ResponseFormatError {
                message: e.to_string(),
                raw_response: raw,
            })?;

        if !parsed.success {
            return Err(LLMError::ProviderError(
                parsed
                    .message
                    .unwrap_or_else(|| "60db text-to-speech request failed".to_string()),
            ));
        }

        let audio_base64 = parsed.audio_base64.ok_or_else(|| {
            LLMError::ResponseFormatError {
                message: "60db response did not contain audio_base64".to_string(),
                raw_response: text_body.clone(),
            }
        })?;

        let audio_data = BASE64
            .decode(audio_base64.as_bytes())
            .map_err(|e| LLMError::ResponseFormatError {
                message: format!("failed to decode base64 audio: {e}"),
                raw_response: text_body,
            })?;

        Ok(audio_data)
    }
}
