// src/backends/bedrock/mod.rs
//! AWS Bedrock backend implementation
//!
//! This module provides integration with AWS Bedrock Runtime API, supporting:
//! - Text completions
//! - Chat completions with tool calls, structured outputs, and vision
//! - Text embeddings

use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::{
    types::{
        CachePointBlock, CachePointType, ContentBlock, ContentBlockDelta, ContentBlockStart,
        ConversationRole, ConverseStreamOutput, JsonSchemaDefinition, Message, OutputConfig,
        OutputFormat, OutputFormatStructure, OutputFormatType, SystemContentBlock, Tool,
        ToolConfiguration, ToolInputSchema, ToolResultBlock, ToolResultContentBlock, ToolUseBlock,
    },
    Client as BedrockClient,
};
use aws_smithy_types::{Blob, Document};
use futures::{Stream, StreamExt};
use serde_json::{json, Value};
use std::collections::{HashMap, VecDeque};
use std::env;
use std::fs;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::OnceCell;

use crate::chat::{
    finish_reason_of, ChatMessage as LlmChatMessage, ChatProvider, StreamChoice,
    StreamChunk as LlmStreamChunk, StreamDelta, StreamResponse, StructuredOutputFormat,
    Tool as LlmTool, ToolChoice as LlmToolChoice,
};
use crate::completion::{
    CompletionProvider, CompletionRequest as GenericCompletionRequest,
    CompletionResponse as GenericCompletionResponse,
};
use crate::embedding::EmbeddingProvider;
use crate::models::ModelsProvider;
use crate::stt::SpeechToTextProvider;
use crate::tts::TextToSpeechProvider;
use crate::{FunctionCall, LLMProvider, ToolCall};

mod error;
mod models;
mod types;

pub use error::{BedrockError, Result};
pub use models::{
    BedrockModel, CrossRegionModel, DirectModel, ModelCapability, ModelCapabilityOverride,
    ModelCapabilityOverrides,
};
pub use types::*;

/// AWS Bedrock backend client
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct BedrockBackend {
    client: Arc<OnceCell<BedrockClient>>,
    region: String,
    // Configuration
    model: Option<BedrockModel>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    timeout_seconds: Option<u64>,
    system: Option<String>,
    top_p: Option<f32>,
    top_k: Option<u32>,
    tools: Option<Vec<LlmTool>>,
    tool_choice: Option<LlmToolChoice>,
    reasoning_effort: Option<String>,
    json_schema: Option<StructuredOutputFormat>,
    model_capability_overrides: Option<ModelCapabilityOverrides>,
}

#[derive(Debug, Clone)]
struct PreparedChatRequest {
    model_id_str: String,
    model: BedrockModel,
    messages: Vec<Message>,
    system: Option<SystemContentBlock>,
    tool_config: Option<ToolConfiguration>,
    inference_config: aws_sdk_bedrockruntime::types::InferenceConfiguration,
    output_config: Option<OutputConfig>,
    additional_model_request_fields: Option<Document>,
}

#[derive(Debug, Default, Clone)]
struct BedrockToolUseState {
    id: String,
    name: String,
    input_buffer: String,
    started: bool,
}

/// One item of a Converse stream: a content chunk, or the token usage Bedrock
/// sends in its metadata once the message has stopped.
#[derive(Debug, Clone)]
enum ConverseEvent {
    Chunk(LlmStreamChunk),
    Usage(UsageInfo),
}

/// Turns one Converse stream output into the events it stands for, in order.
/// Tool inputs are assembled across deltas; a message stop closes any tool
/// use still open before it reports its reason.
fn converse_events(
    output: ConverseStreamOutput,
    tool_states: &mut HashMap<usize, BedrockToolUseState>,
    pending: &mut VecDeque<ConverseEvent>,
) {
    match output {
        ConverseStreamOutput::ContentBlockStart(start) => {
            if let Some(ContentBlockStart::ToolUse(tool_use)) = start.start {
                let index = usize::try_from(start.content_block_index).unwrap_or(0);
                let state = tool_states.entry(index).or_default();
                state.id = tool_use.tool_use_id().to_string();
                state.name = tool_use.name().to_string();
                if !state.started {
                    state.started = true;
                    pending.push_back(ConverseEvent::Chunk(LlmStreamChunk::ToolUseStart {
                        index,
                        id: state.id.clone(),
                        name: state.name.clone(),
                    }));
                }
            }
        }
        ConverseStreamOutput::ContentBlockDelta(delta) => match delta.delta {
            Some(ContentBlockDelta::Text(text)) => {
                if !text.is_empty() {
                    pending.push_back(ConverseEvent::Chunk(LlmStreamChunk::Text(text)));
                }
            }
            Some(ContentBlockDelta::ToolUse(tool_use)) => {
                let index = usize::try_from(delta.content_block_index).unwrap_or(0);
                let state = tool_states.entry(index).or_default();
                if !tool_use.input.is_empty() {
                    state.input_buffer.push_str(&tool_use.input);
                    pending.push_back(ConverseEvent::Chunk(LlmStreamChunk::ToolUseInputDelta {
                        index,
                        partial_json: tool_use.input,
                    }));
                }
            }
            // Reasoning and other block kinds are not content.
            _ => {}
        },
        ConverseStreamOutput::ContentBlockStop(stop) => {
            let index = usize::try_from(stop.content_block_index).unwrap_or(0);
            if let Some(state) = tool_states.remove(&index) {
                if state.started {
                    pending.push_back(ConverseEvent::Chunk(LlmStreamChunk::ToolUseComplete {
                        index,
                        tool_call: state.to_tool_call(),
                    }));
                }
            }
        }
        ConverseStreamOutput::MessageStop(stop) => {
            finish_tool_uses(tool_states, pending);
            pending.push_back(ConverseEvent::Chunk(LlmStreamChunk::Done {
                stop_reason: stop.stop_reason.as_str().to_string(),
            }));
        }
        ConverseStreamOutput::Metadata(metadata) => {
            if let Some(usage) = metadata.usage() {
                pending.push_back(ConverseEvent::Usage(UsageInfo {
                    input_tokens: usage.input_tokens() as u64,
                    output_tokens: usage.output_tokens() as u64,
                    total_tokens: (usage.input_tokens() + usage.output_tokens()) as u64,
                }));
            }
        }
        _ => {}
    }
}

/// Completes every tool use still open, in no particular order.
fn finish_tool_uses(
    tool_states: &mut HashMap<usize, BedrockToolUseState>,
    pending: &mut VecDeque<ConverseEvent>,
) {
    for (index, state) in tool_states.drain() {
        if state.started {
            pending.push_back(ConverseEvent::Chunk(LlmStreamChunk::ToolUseComplete {
                index,
                tool_call: state.to_tool_call(),
            }));
        }
    }
}

/// The OpenAI-shaped chunk for one Converse event: text and complete tool
/// calls as deltas, the usage as a final chunk with no choices, nothing for
/// the rest.
fn struct_response(event: ConverseEvent) -> Option<StreamResponse> {
    match event {
        ConverseEvent::Chunk(LlmStreamChunk::Text(text)) => Some(StreamResponse {
            choices: vec![StreamChoice {
                delta: StreamDelta {
                    content: Some(text),
                    tool_calls: None,
                },
                finish_reason: None,
            }],
            usage: None,
        }),
        ConverseEvent::Chunk(LlmStreamChunk::ToolUseComplete { tool_call, .. }) => {
            Some(StreamResponse {
                choices: vec![StreamChoice {
                    delta: StreamDelta {
                        content: None,
                        tool_calls: Some(vec![tool_call]),
                    },
                    finish_reason: None,
                }],
                usage: None,
            })
        }
        ConverseEvent::Chunk(LlmStreamChunk::Done { stop_reason }) => Some(StreamResponse {
            choices: vec![StreamChoice {
                delta: StreamDelta {
                    content: None,
                    tool_calls: None,
                },
                finish_reason: Some(finish_reason_of(&stop_reason)),
            }],
            usage: None,
        }),
        ConverseEvent::Usage(usage) => Some(StreamResponse {
            choices: vec![],
            usage: Some(crate::chat::Usage {
                prompt_tokens: usage.input_tokens as u32,
                completion_tokens: usage.output_tokens as u32,
                total_tokens: usage.total_tokens as u32,
                completion_tokens_details: None,
                prompt_tokens_details: None,
            }),
        }),
        ConverseEvent::Chunk(_) => None,
    }
}

impl BedrockToolUseState {
    fn to_tool_call(&self) -> ToolCall {
        let arguments = if self.input_buffer.is_empty() {
            "{}".to_string()
        } else {
            self.input_buffer.clone()
        };
        ToolCall {
            id: self.id.clone(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: self.name.clone(),
                arguments,
            },
        }
    }
}

impl BedrockBackend {
    /// Create a new Bedrock backend from environment variables (async)
    pub async fn from_env() -> Result<Self> {
        let config = aws_config::load_defaults(BehaviorVersion::latest()).await;
        let region = config
            .region()
            .map(|r| r.to_string())
            .unwrap_or_else(|| "us-east-1".to_string());
        let client = BedrockClient::new(&config);
        let cell = OnceCell::new();
        cell.set(client).ok();

        Ok(Self {
            client: Arc::new(cell),
            region,
            model: Some(BedrockModel::Direct(DirectModel::ClaudeSonnet4)),
            max_tokens: None,
            temperature: None,
            timeout_seconds: None,
            system: None,
            top_p: None,
            top_k: None,
            tools: None,
            tool_choice: None,
            reasoning_effort: None,
            json_schema: None,
            model_capability_overrides: Self::load_model_capability_overrides()?,
        })
    }

    /// Create a new Bedrock backend with custom configuration (async)
    pub async fn with_config(config: aws_config::SdkConfig) -> Result<Self> {
        let region = config
            .region()
            .map(|r| r.to_string())
            .unwrap_or_else(|| "us-east-1".to_string());
        let client = BedrockClient::new(&config);
        let cell = OnceCell::new();
        cell.set(client).ok();

        Ok(Self {
            client: Arc::new(cell),
            region,
            model: Some(BedrockModel::Direct(DirectModel::ClaudeSonnet4)),
            max_tokens: None,
            temperature: None,
            timeout_seconds: None,
            system: None,
            top_p: None,
            top_k: None,
            tools: None,
            tool_choice: None,
            reasoning_effort: None,
            json_schema: None,
            model_capability_overrides: Self::load_model_capability_overrides()?,
        })
    }

    /// Create a new Bedrock backend with specific options (synchronous, for builder)
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        region: String,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        system: Option<String>,
        top_p: Option<f32>,
        top_k: Option<u32>,
        tools: Option<Vec<LlmTool>>,
        tool_choice: Option<LlmToolChoice>,
        reasoning_effort: Option<String>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Self> {
        Ok(Self {
            client: Arc::new(OnceCell::new()),
            region,
            model: model.map(BedrockModel::from_id),
            max_tokens,
            temperature,
            timeout_seconds,
            system,
            top_p,
            top_k,
            tools,
            tool_choice,
            reasoning_effort,
            json_schema,
            model_capability_overrides: Self::load_model_capability_overrides()?,
        })
    }

    async fn get_client(&self) -> Result<&BedrockClient> {
        self.client
            .get_or_try_init(|| async {
                let config = aws_config::defaults(BehaviorVersion::latest())
                    .region(aws_config::Region::new(self.region.clone()))
                    .load()
                    .await;
                Ok(BedrockClient::new(&config))
            })
            .await
    }

    /// Set the default model
    pub fn with_model(mut self, model: BedrockModel) -> Self {
        self.model = Some(model);
        self
    }

    /// Set the JSON schema for structured output
    pub fn with_json_schema(mut self, schema: StructuredOutputFormat) -> Self {
        self.json_schema = Some(schema);
        self
    }

    /// Override model capability checks (tool use, vision, embeddings, etc.)
    pub fn with_model_capability_overrides(mut self, overrides: ModelCapabilityOverrides) -> Self {
        self.model_capability_overrides = Some(overrides);
        self
    }

    /// Get the AWS region
    pub fn region(&self) -> &str {
        &self.region
    }

    /// Complete a prompt using the Bedrock Converse API
    pub async fn complete_request(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let client = self.get_client().await?;
        let default_model = self
            .model
            .clone()
            .unwrap_or(BedrockModel::Direct(DirectModel::ClaudeSonnet4));
        let model_id = request.model.unwrap_or(default_model);

        // Convert prompt to message format
        let messages = vec![Message::builder()
            .role(ConversationRole::User)
            .content(ContentBlock::Text(request.prompt))
            .build()
            .map_err(|e| BedrockError::InvalidRequest(e.to_string()))?];

        let mut converse_request = client
            .converse()
            .model_id(model_id.model_id())
            .set_messages(Some(messages));

        // Add system prompt if provided
        if let Some(system) = request.system.or(self.system.clone()) {
            converse_request = converse_request.system(SystemContentBlock::Text(system));
        }

        // Add inference configuration
        converse_request = converse_request.inference_config(
            aws_sdk_bedrockruntime::types::InferenceConfiguration::builder()
                .set_max_tokens(request.max_tokens.or(self.max_tokens).map(|t| t as i32))
                .set_temperature(request.temperature.map(|t| t as f32).or(self.temperature))
                .set_top_p(request.top_p.map(|p| p as f32).or(self.top_p))
                .set_stop_sequences(request.stop_sequences)
                .build(),
        );

        let response = converse_request
            .send()
            .await
            .map_err(|e| BedrockError::ApiError(format!("{:?}", e)))?;

        // Extract text from response
        let output = response
            .output()
            .ok_or_else(|| BedrockError::InvalidResponse("No output in response".to_string()))?;

        let text = match output {
            aws_sdk_bedrockruntime::types::ConverseOutput::Message(msg) => msg
                .content()
                .first()
                .and_then(|block| {
                    if let ContentBlock::Text(t) = block {
                        Some(t.clone())
                    } else {
                        None
                    }
                })
                .ok_or_else(|| {
                    BedrockError::InvalidResponse("No text content in response".to_string())
                })?,
            _ => {
                return Err(BedrockError::InvalidResponse(
                    "Unexpected output type".to_string(),
                ))
            }
        };

        let usage = response.usage();
        let finish_reason = Some(format!("{:?}", response.stop_reason));

        Ok(CompletionResponse {
            text,
            model: model_id,
            usage: usage.map(|u| UsageInfo {
                input_tokens: u.input_tokens() as u64,
                output_tokens: u.output_tokens() as u64,
                total_tokens: (u.input_tokens() + u.output_tokens()) as u64,
            }),
            finish_reason,
        })
    }

    /// Chat with the model using the Converse API
    pub async fn chat_request(&self, request: ChatRequest) -> Result<ChatResponse> {
        let client = self.get_client().await?;
        let PreparedChatRequest {
            model_id_str,
            model: model_id,
            messages,
            system,
            tool_config,
            inference_config,
            output_config,
            additional_model_request_fields,
        } = self.prepare_chat_request(request)?;

        let mut converse_request = client
            .converse()
            .model_id(model_id_str)
            .set_messages(Some(messages));

        // Add system prompt if provided
        if let Some(system) = system {
            converse_request = converse_request.system(system);
        }

        // Add tools if provided
        if let Some(tool_config) = tool_config {
            converse_request = converse_request.tool_config(tool_config);
        }

        // Apply native structured output config for models that support outputConfig.textFormat
        // (e.g. Nova). For other models this will be None and the tool-based path is used instead.
        if let Some(output_config) = output_config {
            converse_request = converse_request.output_config(output_config);
        }

        converse_request =
            converse_request.set_additional_model_request_fields(additional_model_request_fields);

        // Add inference configuration
        converse_request = converse_request.inference_config(inference_config);

        let response = converse_request
            .send()
            .await
            .map_err(|e| BedrockError::ApiError(format!("{:?}", e)))?;

        // Convert response
        self.convert_chat_response(response, model_id)
    }

    /// Generate embeddings for text
    pub async fn embed_request(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let client = self.get_client().await?;
        let model_id = request
            .model
            .or(self.model.clone())
            .unwrap_or(BedrockModel::Direct(DirectModel::TitanEmbedV2));

        if !self.model_supports(&model_id, ModelCapability::Embeddings) {
            return Err(BedrockError::UnsupportedOperation(format!(
                "Model {} does not support embeddings",
                model_id.model_id()
            )));
        }

        // Different embedding models have different input formats
        let input_body = match &model_id {
            BedrockModel::Direct(DirectModel::TitanEmbedV2) => {
                json!({
                    "inputText": request.input,
                    "dimensions": request.dimensions.unwrap_or(1024),
                    "normalize": request.normalize.unwrap_or(true),
                })
            }
            BedrockModel::Direct(DirectModel::CohereEmbedV3) => {
                json!({
                    "texts": [request.input],
                    "input_type": request.input_type.unwrap_or_else(|| "search_document".to_string()),
                    "embedding_types": ["float"],
                })
            }
            BedrockModel::Direct(DirectModel::CohereEmbedMultilingualV3) => {
                json!({
                    "texts": [request.input],
                    "input_type": request.input_type.unwrap_or_else(|| "search_document".to_string()),
                    "embedding_types": ["float"],
                })
            }
            BedrockModel::CrossRegion {
                model: models::CrossRegionModel::CohereEmbedV4,
                ..
            } => {
                json!({
                    "texts": [request.input],
                    "input_type": request.input_type.unwrap_or_else(|| "search_document".to_string()),
                    "embedding_types": ["float"],
                })
            }
            _ => {
                return Err(BedrockError::UnsupportedOperation(format!(
                    "Model {} is not an embedding model",
                    model_id.model_id()
                )));
            }
        };

        let response = client
            .invoke_model()
            .model_id(model_id.model_id())
            .body(Blob::new(serde_json::to_vec(&input_body)?))
            .send()
            .await
            .map_err(|e| BedrockError::ApiError(format!("{:?}", e)))?;

        let body: Value = serde_json::from_slice(response.body().as_ref())?;

        let embedding = match &model_id {
            BedrockModel::Direct(DirectModel::TitanEmbedV2) => body
                .get("embedding")
                .and_then(|e| e.as_array())
                .ok_or_else(|| {
                    BedrockError::InvalidResponse("No embedding in response".to_string())
                })?
                .iter()
                .filter_map(|v| v.as_f64())
                .collect(),
            BedrockModel::Direct(DirectModel::CohereEmbedV3) => body
                .get("embeddings")
                .and_then(|e| e.get("float"))
                .and_then(|e| e.as_array())
                .and_then(|arr| arr.first())
                .and_then(|e| e.as_array())
                .ok_or_else(|| {
                    BedrockError::InvalidResponse("No embeddings in response".to_string())
                })?
                .iter()
                .filter_map(|v| v.as_f64())
                .collect(),
            BedrockModel::Direct(DirectModel::CohereEmbedMultilingualV3) => body
                .get("embeddings")
                .and_then(|e| e.get("float"))
                .and_then(|e| e.as_array())
                .and_then(|arr| arr.first())
                .and_then(|e| e.as_array())
                .ok_or_else(|| {
                    BedrockError::InvalidResponse("No embeddings in response".to_string())
                })?
                .iter()
                .filter_map(|v| v.as_f64())
                .collect(),
            BedrockModel::CrossRegion {
                model: models::CrossRegionModel::CohereEmbedV4,
                ..
            } => body
                .get("embeddings")
                .and_then(|e| e.get("float"))
                .and_then(|e| e.as_array())
                .and_then(|arr| arr.first())
                .and_then(|e| e.as_array())
                .ok_or_else(|| {
                    BedrockError::InvalidResponse("No embeddings in response".to_string())
                })?
                .iter()
                .filter_map(|v| v.as_f64())
                .collect(),
            _ => vec![],
        };

        let dimensions = embedding.len();

        Ok(EmbeddingResponse {
            embedding,
            model: model_id,
            dimensions,
        })
    }

    /// Embed 1..=96 texts in one Bedrock call. Only Cohere embed models accept batches.
    pub async fn embed_batch_request(
        &self,
        request: EmbeddingBatchRequest,
    ) -> Result<EmbeddingBatchResponse> {
        let model_id = request
            .model
            .clone()
            .or(self.model.clone())
            .unwrap_or(BedrockModel::Direct(DirectModel::TitanEmbedV2));

        if !self.model_supports(&model_id, ModelCapability::Embeddings) {
            return Err(BedrockError::UnsupportedOperation(format!(
                "Model {} does not support embeddings",
                model_id.model_id()
            )));
        }
        if !matches!(
            model_id,
            BedrockModel::Direct(DirectModel::CohereEmbedV3)
                | BedrockModel::Direct(DirectModel::CohereEmbedMultilingualV3)
                | BedrockModel::CrossRegion {
                    model: models::CrossRegionModel::CohereEmbedV4,
                    ..
                }
        ) {
            return Err(BedrockError::UnsupportedOperation(format!(
                "Model {} does not support batched embeddings",
                model_id.model_id()
            )));
        }
        let expected = request.inputs.len();
        let input_body = embed_batch_body(request.inputs, request.input_type)?;

        let client = self.get_client().await?;
        let response = client
            .invoke_model()
            .model_id(model_id.model_id())
            .body(Blob::new(serde_json::to_vec(&input_body)?))
            .send()
            .await
            .map_err(|e| BedrockError::ApiError(format!("{:?}", e)))?;

        let body: Value = serde_json::from_slice(response.body().as_ref())?;
        let embeddings: Vec<Vec<f64>> = body
            .get("embeddings")
            .and_then(|e| e.get("float"))
            .and_then(|e| e.as_array())
            .ok_or_else(|| BedrockError::InvalidResponse("No embeddings in response".to_string()))?
            .iter()
            .map(|embedding| {
                embedding
                    .as_array()
                    .map(|values| values.iter().filter_map(|v| v.as_f64()).collect())
                    .ok_or_else(|| {
                        BedrockError::InvalidResponse("Embedding is not an array".to_string())
                    })
            })
            .collect::<Result<_>>()?;
        if embeddings.len() != expected {
            return Err(BedrockError::InvalidResponse(format!(
                "Expected {} embeddings, got {}",
                expected,
                embeddings.len()
            )));
        }

        let dimensions = embeddings.first().map_or(0, Vec::len);
        Ok(EmbeddingBatchResponse {
            embeddings,
            model: model_id,
            dimensions,
        })
    }

    /// Rank 1..=1000 documents against a query with a rerank model (Cohere Rerank 3.5).
    pub async fn rerank_request(&self, request: RerankRequest) -> Result<RerankResponse> {
        let model_id = request
            .model
            .clone()
            .unwrap_or(BedrockModel::Direct(DirectModel::CohereRerankV35));

        if !self.model_supports(&model_id, ModelCapability::Rerank) {
            return Err(BedrockError::UnsupportedOperation(format!(
                "Model {} does not support rerank",
                model_id.model_id()
            )));
        }
        let input_body = rerank_body(request.query, request.documents, request.top_n)?;

        let client = self.get_client().await?;
        let response = client
            .invoke_model()
            .model_id(model_id.model_id())
            .body(Blob::new(serde_json::to_vec(&input_body)?))
            .send()
            .await
            .map_err(|e| BedrockError::ApiError(format!("{:?}", e)))?;

        let body: Value = serde_json::from_slice(response.body().as_ref())?;
        let results = body
            .get("results")
            .cloned()
            .ok_or_else(|| BedrockError::InvalidResponse("No results in response".to_string()))?;
        let results: Vec<RerankResult> = serde_json::from_value(results)?;

        Ok(RerankResponse {
            results,
            model: model_id,
        })
    }

    /// Stream chat responses
    pub async fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> Result<impl futures::Stream<Item = Result<ChatStreamChunk>>> {
        let client = self.get_client().await?;
        let PreparedChatRequest {
            model_id_str,
            model: _,
            messages,
            system,
            tool_config,
            inference_config,
            output_config,
            additional_model_request_fields,
        } = self.prepare_chat_request(request)?;

        let mut converse_request = client
            .converse_stream()
            .model_id(model_id_str)
            .set_messages(Some(messages));

        // Add system prompt if provided
        if let Some(system) = system {
            converse_request = converse_request.system(system);
        }

        // Add tools if provided
        if let Some(tool_config) = tool_config {
            converse_request = converse_request.tool_config(tool_config);
        }

        // Apply native structured output config for models that support outputConfig.textFormat
        if let Some(output_config) = output_config {
            converse_request = converse_request.output_config(output_config);
        }

        converse_request =
            converse_request.set_additional_model_request_fields(additional_model_request_fields);

        // Add inference configuration
        converse_request = converse_request.inference_config(inference_config);

        let response = converse_request
            .send()
            .await
            .map_err(|e| BedrockError::ApiError(format!("{:?}", e)))?;

        let stream = response.stream;

        Ok(futures::stream::unfold(stream, |mut stream| async move {
            loop {
                let next_item = stream.recv().await;
                match next_item {
                    Ok(Some(event)) => {
                        let chunk = match event {
                            ConverseStreamOutput::ContentBlockStart(_) => {
                                continue;
                            }
                            ConverseStreamOutput::ContentBlockDelta(delta) => match delta.delta {
                                Some(ContentBlockDelta::Text(text)) => Some(ChatStreamChunk {
                                    delta: text,
                                    finish_reason: None,
                                }),
                                Some(ContentBlockDelta::ToolUse(tool_use)) => {
                                    Some(ChatStreamChunk {
                                        delta: tool_use.input,
                                        finish_reason: None,
                                    })
                                }
                                _ => continue,
                            },
                            ConverseStreamOutput::ContentBlockStop(_) => {
                                continue;
                            }
                            ConverseStreamOutput::MessageStart(_) => {
                                continue;
                            }
                            ConverseStreamOutput::MessageStop(stop) => {
                                let finish_reason = Some(stop.stop_reason.as_str().to_string());
                                Some(ChatStreamChunk {
                                    delta: String::new(),
                                    finish_reason,
                                })
                            }
                            ConverseStreamOutput::Metadata(_) => {
                                continue;
                            }
                            _ => continue,
                        };

                        return chunk.map(|c| (Ok(c), stream));
                    }
                    Ok(None) => return None,
                    Err(e) => {
                        return Some((Err(BedrockError::StreamError(format!("{:?}", e))), stream))
                    }
                }
            }
        }))
    }

    /// Stream chat responses with tool call events.
    pub async fn chat_stream_with_tools(
        &self,
        request: ChatRequest,
    ) -> Result<impl futures::Stream<Item = Result<LlmStreamChunk>>> {
        let events = self.converse_stream(request).await?;
        Ok(events.filter_map(|item| async move {
            match item {
                Ok(ConverseEvent::Chunk(chunk)) => Some(Ok(chunk)),
                Ok(ConverseEvent::Usage(_)) => None,
                Err(error) => Some(Err(error)),
            }
        }))
    }

    /// The Converse stream as content chunks and, after the message stops,
    /// the turn's token usage.
    async fn converse_stream(
        &self,
        request: ChatRequest,
    ) -> Result<impl futures::Stream<Item = Result<ConverseEvent>>> {
        let client = self.get_client().await?;
        let PreparedChatRequest {
            model_id_str,
            model: _,
            messages,
            system,
            tool_config,
            inference_config,
            output_config,
            additional_model_request_fields,
        } = self.prepare_chat_request(request)?;

        let mut converse_request = client
            .converse_stream()
            .model_id(model_id_str)
            .set_messages(Some(messages));

        if let Some(system) = system {
            converse_request = converse_request.system(system);
        }

        if let Some(tool_config) = tool_config {
            converse_request = converse_request.tool_config(tool_config);
        }

        // Apply native structured output config for models that support outputConfig.textFormat
        if let Some(output_config) = output_config {
            converse_request = converse_request.output_config(output_config);
        }

        converse_request =
            converse_request.set_additional_model_request_fields(additional_model_request_fields);

        converse_request = converse_request.inference_config(inference_config);

        let response = converse_request
            .send()
            .await
            .map_err(|e| BedrockError::ApiError(format!("{:?}", e)))?;

        let stream = response.stream;

        let initial_state = (
            stream,
            HashMap::<usize, BedrockToolUseState>::new(),
            VecDeque::<ConverseEvent>::new(),
        );

        Ok(futures::stream::unfold(
            initial_state,
            |(mut stream, mut tool_states, mut pending)| async move {
                loop {
                    if let Some(event) = pending.pop_front() {
                        return Some((Ok(event), (stream, tool_states, pending)));
                    }

                    match stream.recv().await {
                        Ok(Some(output)) => {
                            converse_events(output, &mut tool_states, &mut pending);
                        }
                        Ok(None) => {
                            finish_tool_uses(&mut tool_states, &mut pending);
                            if let Some(event) = pending.pop_front() {
                                return Some((Ok(event), (stream, tool_states, pending)));
                            }
                            return None;
                        }
                        Err(e) => {
                            return Some((
                                Err(BedrockError::StreamError(format!("{:?}", e))),
                                (stream, tool_states, pending),
                            ))
                        }
                    }
                }
            },
        ))
    }

    // Helper methods

    fn prepare_chat_request(&self, request: ChatRequest) -> Result<PreparedChatRequest> {
        let default_model = self
            .model
            .clone()
            .unwrap_or(BedrockModel::Direct(DirectModel::ClaudeSonnet4));
        let model_id = request.model.unwrap_or(default_model);

        // Validate model capabilities
        if !self.model_supports(&model_id, ModelCapability::Chat) {
            return Err(BedrockError::UnsupportedOperation(format!(
                "Model {} does not support chat",
                model_id.model_id()
            )));
        }

        // Convert messages
        let mut system_from_messages: Option<String> = None;
        let mut converted_messages: Vec<Message> = Vec::new();

        for msg in &request.messages {
            if msg.role == "system" {
                // Only accept the first system message we encounter. Ignore
                // subsequent system messages ("first one wins"). Also prefer a
                // plain text part when a multimodal system message is used.
                if system_from_messages.is_none() {
                    match &msg.content {
                        MessageContent::Text(t) => {
                            system_from_messages = Some(t.clone());
                        }
                        MessageContent::MultiModal(parts) => {
                            for part in parts {
                                if let ContentPart::Text { text } = part {
                                    system_from_messages = Some(text.clone());
                                    break;
                                }
                            }
                        }
                    }
                }

                // skip adding this message to the converted_messages
                continue;
            }

            // Non-system messages are converted normally
            converted_messages.push(self.convert_message(msg)?);
        }

        let messages = converted_messages;

        // System prompt: prefer explicit request.system, then backend default,
        // then any system text found inside request.messages
        let system = request
            .system
            .or(self.system.clone())
            .or(system_from_messages)
            .map(SystemContentBlock::Text);

        // Tools
        let mut bedrock_tools = Vec::new();

        // Check if any tool has cache_control before consuming request.tools
        let request_tools_need_cache = request
            .tools
            .as_ref()
            .map(|tools| tools.iter().any(|t| t.cache_control.is_some()))
            .unwrap_or(false);

        if let Some(tools) = request.tools {
            for tool in tools {
                bedrock_tools.push(self.convert_tool(&tool)?);
            }
        }

        let mut tool_choice = self.tool_choice.clone();
        let mut output_config: Option<OutputConfig> = None;

        if let Some(response_format) = self.json_schema.as_ref() {
            let schema = response_format.schema.clone().ok_or_else(|| {
                BedrockError::InvalidRequest(
                    "Structured output format must contain a schema".to_string(),
                )
            })?;

            if self.model_supports(&model_id, ModelCapability::NativeStructuredOutput) {
                // Nova and other models that advertise NativeStructuredOutput receive the schema
                // via outputConfig.textFormat. The response arrives as ContentBlock::Text
                // (plain JSON), so no tool-call unwrapping is needed.
                let schema_str = serde_json::to_string(&schema).map_err(|e| {
                    BedrockError::InvalidRequest(format!("Failed to serialize schema: {}", e))
                })?;

                let mut json_schema_builder = JsonSchemaDefinition::builder().schema(schema_str);
                if !response_format.name.is_empty() {
                    json_schema_builder = json_schema_builder.name(&response_format.name);
                }
                if let Some(desc) = &response_format.description {
                    json_schema_builder = json_schema_builder.description(desc);
                }

                let json_schema_def = json_schema_builder.build().map_err(|e| {
                    BedrockError::InvalidRequest(format!(
                        "Failed to build JSON schema definition: {:?}",
                        e
                    ))
                })?;

                let output_format = OutputFormat::builder()
                    .r#type(OutputFormatType::JsonSchema)
                    .structure(OutputFormatStructure::JsonSchema(json_schema_def))
                    .build()
                    .map_err(|e| {
                        BedrockError::InvalidRequest(format!(
                            "Failed to build output format: {:?}",
                            e
                        ))
                    })?;

                // OutputConfig::builder().build() is infallible - the SDK builder has no
                // required fields beyond what we set via .text_format().
                output_config = Some(OutputConfig::builder().text_format(output_format).build());
            } else {
                // Fallback for models without native structured output (e.g. Claude):
                // inject a synthetic tool and force the model to call it, then unwrap
                // the tool call in convert_chat_response.
                let input_schema = ToolInputSchema::Json(Self::value_to_document(&schema));

                let tool_spec = aws_sdk_bedrockruntime::types::ToolSpecification::builder()
                    .name("json_schema_tool")
                    .description(
                        "Generates structured output in JSON format according to the provided schema.",
                    )
                    .input_schema(input_schema)
                    .build()
                    .map_err(|e| {
                        BedrockError::InvalidRequest(format!("Failed to build tool spec: {:?}", e))
                    })?;

                bedrock_tools.push(Tool::ToolSpec(tool_spec));
                tool_choice = Some(LlmToolChoice::Tool("json_schema_tool".to_string()));
            }
        }

        if let Some(tools) = &self.tools {
            for tool in tools {
                bedrock_tools.push(self.convert_llm_tool(tool)?);
            }
        }

        // Append a CachePoint if any tool has cache_control set
        let self_tools_need_cache = self
            .tools
            .as_ref()
            .map(|tools| tools.iter().any(|t| t.cache_control.is_some()))
            .unwrap_or(false);

        if (request_tools_need_cache || self_tools_need_cache) && !bedrock_tools.is_empty() {
            bedrock_tools.push(Tool::CachePoint(
                CachePointBlock::builder()
                    .r#type(CachePointType::Default)
                    .build()
                    .map_err(|e| {
                        BedrockError::InvalidRequest(format!(
                            "Failed to build cache point: {:?}",
                            e
                        ))
                    })?,
            ));
        }

        let effective_tool_choice = tool_choice.unwrap_or(LlmToolChoice::Auto);
        let tools_disabled = matches!(effective_tool_choice, LlmToolChoice::None);
        let mut tool_config = None;

        if !bedrock_tools.is_empty() && !matches!(effective_tool_choice, LlmToolChoice::None) {
            if !self.model_supports(&model_id, ModelCapability::ToolUse) {
                return Err(BedrockError::UnsupportedOperation(format!(
                    "Model {} does not support tool use",
                    model_id.model_id()
                )));
            }

            let aws_tool_choice = match effective_tool_choice {
                LlmToolChoice::Auto => Some(aws_sdk_bedrockruntime::types::ToolChoice::Auto(
                    aws_sdk_bedrockruntime::types::AutoToolChoice::builder().build(),
                )),
                LlmToolChoice::Any => Some(aws_sdk_bedrockruntime::types::ToolChoice::Any(
                    aws_sdk_bedrockruntime::types::AnyToolChoice::builder().build(),
                )),
                LlmToolChoice::Tool(name) => Some(aws_sdk_bedrockruntime::types::ToolChoice::Tool(
                    aws_sdk_bedrockruntime::types::SpecificToolChoice::builder()
                        .name(name)
                        .build()
                        .map_err(|e| {
                            BedrockError::InvalidRequest(format!(
                                "Failed to build specific tool choice: {:?}",
                                e
                            ))
                        })?,
                )),
                LlmToolChoice::None => None,
            };

            tool_config = Some(
                ToolConfiguration::builder()
                    .set_tools(Some(bedrock_tools))
                    .set_tool_choice(aws_tool_choice)
                    .build()
                    .map_err(|e| BedrockError::InvalidRequest(e.to_string()))?,
            );
        }

        let mut additional_fields = serde_json::Map::new();
        if tool_config.is_none() && has_tool_blocks(&messages) {
            if tools_disabled {
                if !is_anthropic_model(&model_id) {
                    return Err(BedrockError::UnsupportedOperation(format!(
                        "Model {} cannot disable tools when the history contains tool calls",
                        model_id.model_id()
                    )));
                }
                tool_config = Some(placeholder_tool_config(true)?);
                additional_fields.insert("tool_choice".to_string(), json!({"type": "none"}));
            } else if self.model_supports(&model_id, ModelCapability::ToolUse) {
                tool_config = Some(placeholder_tool_config(false)?);
            }
        }

        if let Some(effort) = self.claude_effort(&model_id, output_config.is_some()) {
            additional_fields.insert("output_config".to_string(), json!({"effort": effort}));
        }
        let additional_model_request_fields = (!additional_fields.is_empty())
            .then(|| Self::value_to_document(&Value::Object(additional_fields)));

        // Inference config
        let inference_config = aws_sdk_bedrockruntime::types::InferenceConfiguration::builder()
            .set_max_tokens(request.max_tokens.or(self.max_tokens).map(|t| t as i32))
            .set_temperature(request.temperature.map(|t| t as f32).or(self.temperature))
            .set_top_p(request.top_p.map(|p| p as f32).or(self.top_p))
            .set_stop_sequences(request.stop_sequences)
            .build();

        Ok(PreparedChatRequest {
            model_id_str: model_id.model_id().to_string(),
            model: model_id,
            messages,
            system,
            tool_config,
            inference_config,
            output_config,
            additional_model_request_fields,
        })
    }

    fn convert_message(&self, msg: &ChatMessage) -> Result<Message> {
        let role = match msg.role.as_str() {
            "user" => ConversationRole::User,
            "assistant" => ConversationRole::Assistant,
            _ => {
                return Err(BedrockError::InvalidRequest(format!(
                    "Invalid role: {}",
                    msg.role
                )))
            }
        };

        let mut message_builder = Message::builder().role(role);

        match &msg.content {
            MessageContent::Text(text) => {
                message_builder = message_builder.content(ContentBlock::Text(text.clone()));
            }
            MessageContent::MultiModal(parts) => {
                for part in parts {
                    match part {
                        ContentPart::Text { text } => {
                            message_builder =
                                message_builder.content(ContentBlock::Text(text.clone()));
                        }
                        ContentPart::Image { source, media_type } => {
                            let image = aws_sdk_bedrockruntime::types::ImageBlock::builder()
                                .format(Self::convert_media_type(media_type)?)
                                .source(aws_sdk_bedrockruntime::types::ImageSource::Bytes(
                                    Blob::new(source.clone()),
                                ))
                                .build()
                                .map_err(|e| BedrockError::InvalidRequest(e.to_string()))?;

                            message_builder = message_builder.content(ContentBlock::Image(image));
                        }
                        ContentPart::ToolUse { id, name, input } => {
                            let tool_use = ToolUseBlock::builder()
                                .tool_use_id(id)
                                .name(name)
                                .input(Document::Object(
                                    input
                                        .as_object()
                                        .ok_or_else(|| {
                                            BedrockError::InvalidRequest(
                                                "Tool input must be an object".to_string(),
                                            )
                                        })?
                                        .iter()
                                        .map(|(k, v)| (k.clone(), Self::value_to_document(v)))
                                        .collect(),
                                ))
                                .build()
                                .map_err(|e| BedrockError::InvalidRequest(e.to_string()))?;

                            message_builder =
                                message_builder.content(ContentBlock::ToolUse(tool_use));
                        }
                        ContentPart::ToolResult {
                            tool_use_id,
                            content,
                            is_error,
                        } => {
                            let result = ToolResultBlock::builder()
                                .tool_use_id(tool_use_id)
                                .content(ToolResultContentBlock::Text(content.clone()))
                                .set_status(if *is_error {
                                    Some(aws_sdk_bedrockruntime::types::ToolResultStatus::Error)
                                } else {
                                    Some(aws_sdk_bedrockruntime::types::ToolResultStatus::Success)
                                })
                                .build()
                                .map_err(|e| BedrockError::InvalidRequest(e.to_string()))?;

                            message_builder =
                                message_builder.content(ContentBlock::ToolResult(result));
                        }
                    }
                }
            }
        }

        message_builder
            .build()
            .map_err(|e| BedrockError::InvalidRequest(e.to_string()))
    }

    fn convert_tool(&self, tool: &ToolDefinition) -> Result<Tool> {
        let input_schema = ToolInputSchema::Json(Document::Object(
            tool.input_schema
                .as_object()
                .ok_or_else(|| {
                    BedrockError::InvalidRequest("Tool input schema must be an object".to_string())
                })?
                .iter()
                .map(|(k, v)| (k.clone(), Self::value_to_document(v)))
                .collect(),
        ));

        let tool_spec = aws_sdk_bedrockruntime::types::ToolSpecification::builder()
            .name(&tool.name)
            .description(&tool.description)
            .input_schema(input_schema)
            .build()
            .map_err(|e| BedrockError::InvalidRequest(e.to_string()))?;

        Ok(Tool::ToolSpec(tool_spec))
    }

    fn convert_chat_response(
        &self,
        response: aws_sdk_bedrockruntime::operation::converse::ConverseOutput,
        model: BedrockModel,
    ) -> Result<ChatResponse> {
        let output = response
            .output()
            .ok_or_else(|| BedrockError::InvalidResponse("No output in response".to_string()))?;

        let message = match output {
            aws_sdk_bedrockruntime::types::ConverseOutput::Message(msg) => {
                let mut content_parts = Vec::new();
                let mut json_schema_output = None;

                for block in msg.content() {
                    match block {
                        ContentBlock::Text(text) => {
                            content_parts.push(ContentPart::Text { text: text.clone() });
                        }
                        ContentBlock::ToolUse(tool_use) => {
                            if self.json_schema.is_some() && tool_use.name() == "json_schema_tool" {
                                let input = Self::document_to_value(&tool_use.input);
                                json_schema_output = Some(input);
                            }

                            content_parts.push(ContentPart::ToolUse {
                                id: tool_use.tool_use_id().to_string(),
                                name: tool_use.name().to_string(),
                                input: Self::document_to_value(&tool_use.input),
                            });
                        }
                        _ => {}
                    }
                }

                if let Some(json_output) = json_schema_output {
                    ChatMessage {
                        role: "assistant".to_string(),
                        content: MessageContent::Text(
                            serde_json::to_string(&json_output).unwrap_or_default(),
                        ),
                    }
                } else {
                    ChatMessage {
                        role: "assistant".to_string(),
                        content: if content_parts.len() == 1 {
                            if let ContentPart::Text { text } = &content_parts[0] {
                                MessageContent::Text(text.clone())
                            } else {
                                MessageContent::MultiModal(content_parts)
                            }
                        } else {
                            MessageContent::MultiModal(content_parts)
                        },
                    }
                }
            }
            _ => {
                return Err(BedrockError::InvalidResponse(
                    "Unexpected output type".to_string(),
                ))
            }
        };

        let usage = response.usage();
        let finish_reason = Some(format!("{:?}", response.stop_reason));

        Ok(ChatResponse {
            message,
            model,
            usage: usage.map(|u| UsageInfo {
                input_tokens: u.input_tokens() as u64,
                output_tokens: u.output_tokens() as u64,
                total_tokens: (u.input_tokens() + u.output_tokens()) as u64,
            }),
            finish_reason,
        })
    }

    fn convert_llm_tool(&self, tool: &LlmTool) -> Result<Tool> {
        if tool.tool_type != "function" {
            return Err(BedrockError::InvalidRequest(format!(
                "Unsupported tool type: {}",
                tool.tool_type
            )));
        }

        let input_schema =
            ToolInputSchema::Json(Self::value_to_document(&tool.function.parameters));

        let tool_spec = aws_sdk_bedrockruntime::types::ToolSpecification::builder()
            .name(&tool.function.name)
            .description(&tool.function.description)
            .input_schema(input_schema)
            .build()
            .map_err(|e| {
                BedrockError::InvalidRequest(format!("Failed to build tool spec: {:?}", e))
            })?;

        Ok(Tool::ToolSpec(tool_spec))
    }

    fn convert_media_type(media_type: &str) -> Result<aws_sdk_bedrockruntime::types::ImageFormat> {
        match media_type {
            "image/png" => Ok(aws_sdk_bedrockruntime::types::ImageFormat::Png),
            "image/jpeg" | "image/jpg" => Ok(aws_sdk_bedrockruntime::types::ImageFormat::Jpeg),
            "image/gif" => Ok(aws_sdk_bedrockruntime::types::ImageFormat::Gif),
            "image/webp" => Ok(aws_sdk_bedrockruntime::types::ImageFormat::Webp),
            _ => Err(BedrockError::InvalidRequest(format!(
                "Unsupported media type: {}",
                media_type
            ))),
        }
    }

    /// The `effort` to send a Claude model through the Messages API fields
    /// Converse passes on (`additionalModelRequestFields.output_config`).
    /// Not sent with Converse's own `outputConfig` (structured output), so the
    /// two never meet in one request, nor to non-Claude models.
    fn claude_effort(&self, model: &BedrockModel, structured: bool) -> Option<&str> {
        let effort = self.reasoning_effort.as_deref()?;
        if !is_anthropic_model(model) {
            return None;
        }
        if structured {
            log::debug!("effort {effort} not sent with Converse structured output");
            return None;
        }
        Some(effort)
    }

    fn value_to_document(value: &Value) -> Document {
        match value {
            Value::Null => Document::Null,
            Value::Bool(b) => Document::Bool(*b),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Document::Number(aws_smithy_types::Number::PosInt(i as u64))
                } else if let Some(f) = n.as_f64() {
                    Document::Number(aws_smithy_types::Number::Float(f))
                } else {
                    Document::Null
                }
            }
            Value::String(s) => Document::String(s.clone()),
            Value::Array(arr) => Document::Array(arr.iter().map(Self::value_to_document).collect()),
            Value::Object(obj) => Document::Object(
                obj.iter()
                    .map(|(k, v)| (k.clone(), Self::value_to_document(v)))
                    .collect(),
            ),
        }
    }

    fn model_supports(&self, model: &BedrockModel, capability: ModelCapability) -> bool {
        if let Some(overrides) = &self.model_capability_overrides {
            if let Some(supports) = overrides.supports(model, capability) {
                return supports;
            }
        }

        model.supports(capability)
    }

    fn load_model_capability_overrides() -> Result<Option<ModelCapabilityOverrides>> {
        if let Ok(path) = env::var("LLM_BEDROCK_MODEL_CAPABILITIES_PATH") {
            let trimmed = path.trim();
            if !trimmed.is_empty() {
                let contents = fs::read_to_string(trimmed).map_err(|e| {
                    BedrockError::ConfigurationError(format!(
                        "Failed to read model capabilities file {}: {}",
                        trimmed, e
                    ))
                })?;
                let overrides = Self::parse_model_capability_overrides(&contents)?;
                return Ok(Some(overrides));
            }
        }

        if let Ok(raw) = env::var("LLM_BEDROCK_MODEL_CAPABILITIES") {
            let trimmed = raw.trim();
            if !trimmed.is_empty() {
                let overrides = Self::parse_model_capability_overrides(trimmed)?;
                return Ok(Some(overrides));
            }
        }

        Ok(None)
    }

    fn parse_model_capability_overrides(contents: &str) -> Result<ModelCapabilityOverrides> {
        if let Ok(config) = serde_json::from_str::<ModelCapabilityOverrides>(contents) {
            return Ok(config);
        }
        if let Ok(map) = serde_json::from_str::<HashMap<String, ModelCapabilityOverride>>(contents)
        {
            return Ok(ModelCapabilityOverrides {
                models: map,
                model: Vec::new(),
            });
        }
        if let Ok(config) = toml::from_str::<ModelCapabilityOverrides>(contents) {
            return Ok(config);
        }
        if let Ok(map) = toml::from_str::<HashMap<String, ModelCapabilityOverride>>(contents) {
            return Ok(ModelCapabilityOverrides {
                models: map,
                model: Vec::new(),
            });
        }
        if let Ok(config) = serde_yaml::from_str::<ModelCapabilityOverrides>(contents) {
            return Ok(config);
        }
        if let Ok(map) = serde_yaml::from_str::<HashMap<String, ModelCapabilityOverride>>(contents)
        {
            return Ok(ModelCapabilityOverrides {
                models: map,
                model: Vec::new(),
            });
        }

        Err(BedrockError::ConfigurationError(
            "Failed to parse model capability overrides (expected JSON, TOML, or YAML)".to_string(),
        ))
    }

    fn document_to_value(doc: &Document) -> Value {
        match doc {
            Document::Null => Value::Null,
            Document::Bool(b) => Value::Bool(*b),
            Document::Number(n) => match n {
                aws_smithy_types::Number::PosInt(i) => json!(*i),
                aws_smithy_types::Number::NegInt(i) => json!(*i),
                aws_smithy_types::Number::Float(f) => json!(*f),
            },
            Document::String(s) => Value::String(s.clone()),
            Document::Array(arr) => Value::Array(arr.iter().map(Self::document_to_value).collect()),
            Document::Object(obj) => Value::Object(
                obj.iter()
                    .map(|(k, v)| (k.clone(), Self::document_to_value(v)))
                    .collect(),
            ),
        }
    }
}

#[async_trait]
impl ModelsProvider for BedrockBackend {
    async fn list_models(
        &self,
        _request: Option<&crate::models::ModelListRequest>,
    ) -> std::result::Result<Box<dyn crate::models::ModelListResponse>, crate::error::LLMError>
    {
        Err(crate::error::LLMError::Generic(
            "List models not supported for Bedrock".to_string(),
        ))
    }
}

#[async_trait]
impl TextToSpeechProvider for BedrockBackend {
    async fn speech(&self, _input: &str) -> std::result::Result<Vec<u8>, crate::error::LLMError> {
        Err(crate::error::LLMError::Generic(
            "TTS not supported for Bedrock".to_string(),
        ))
    }
}

#[async_trait]
impl SpeechToTextProvider for BedrockBackend {
    async fn transcribe(
        &self,
        _audio: Vec<u8>,
    ) -> std::result::Result<String, crate::error::LLMError> {
        Err(crate::error::LLMError::Generic(
            "STT not supported for Bedrock".to_string(),
        ))
    }
}

const AUDIO_UNSUPPORTED: &str = "Audio messages are not supported by AWS Bedrock chat";

#[async_trait]
impl ChatProvider for BedrockBackend {
    async fn chat_with_tools(
        &self,
        messages: &[LlmChatMessage],
        tools: Option<&[LlmTool]>,
    ) -> std::result::Result<Box<dyn crate::chat::ChatResponse>, crate::error::LLMError> {
        crate::chat::ensure_no_audio(messages, AUDIO_UNSUPPORTED)?;
        let aws_messages = history_to_messages(messages);

        let mut request = ChatRequest::new(aws_messages);

        if let Some(tools) = tools {
            let tool_defs: Vec<ToolDefinition> = tools
                .iter()
                .map(|t| ToolDefinition {
                    name: t.function.name.clone(),
                    description: t.function.description.clone(),
                    input_schema: t.function.parameters.clone(),
                    cache_control: t.cache_control.clone(),
                })
                .collect();

            request = request.with_tools(tool_defs);
        }

        let response = self
            .chat_request(request)
            .await
            .map_err(|e| crate::error::LLMError::ProviderError(e.to_string()))?;
        Ok(Box::new(response))
    }

    async fn chat_stream(
        &self,
        messages: &[LlmChatMessage],
    ) -> std::result::Result<
        Pin<Box<dyn Stream<Item = std::result::Result<String, crate::error::LLMError>> + Send>>,
        crate::error::LLMError,
    > {
        let aws_messages = history_to_messages(messages);

        let request = ChatRequest::new(aws_messages);
        let stream = self
            .chat_stream(request)
            .await
            .map_err(|e| crate::error::LLMError::ProviderError(e.to_string()))?;

        let stream = stream.map(|item| match item {
            Ok(chunk) => Ok(chunk.delta),
            Err(e) => Err(crate::error::LLMError::ProviderError(e.to_string())),
        });

        Ok(Box::pin(stream))
    }

    async fn chat_stream_struct(
        &self,
        messages: &[LlmChatMessage],
    ) -> std::result::Result<
        Pin<
            Box<
                dyn Stream<Item = std::result::Result<StreamResponse, crate::error::LLMError>>
                    + Send,
            >,
        >,
        crate::error::LLMError,
    > {
        let aws_messages = history_to_messages(messages);

        let request = ChatRequest::new(aws_messages);
        let stream = self
            .converse_stream(request)
            .await
            .map_err(|e| crate::error::LLMError::ProviderError(e.to_string()))?;

        let stream = stream.filter_map(|item| async move {
            match item {
                Ok(event) => struct_response(event).map(Ok),
                Err(e) => Some(Err(crate::error::LLMError::ProviderError(e.to_string()))),
            }
        });

        Ok(Box::pin(stream))
    }

    async fn chat_stream_with_tools(
        &self,
        messages: &[LlmChatMessage],
        tools: Option<&[LlmTool]>,
    ) -> std::result::Result<
        Pin<
            Box<
                dyn Stream<Item = std::result::Result<LlmStreamChunk, crate::error::LLMError>>
                    + Send,
            >,
        >,
        crate::error::LLMError,
    > {
        let aws_messages = history_to_messages(messages);

        let mut request = ChatRequest::new(aws_messages);

        if let Some(tools) = tools {
            let tool_defs: Vec<ToolDefinition> = tools
                .iter()
                .map(|t| ToolDefinition {
                    name: t.function.name.clone(),
                    description: t.function.description.clone(),
                    input_schema: t.function.parameters.clone(),
                    cache_control: t.cache_control.clone(),
                })
                .collect();

            request = request.with_tools(tool_defs);
        }

        let stream = BedrockBackend::chat_stream_with_tools(self, request)
            .await
            .map_err(|e| crate::error::LLMError::ProviderError(e.to_string()))?;

        let stream = stream.map(|item| match item {
            Ok(chunk) => Ok(chunk),
            Err(e) => Err(crate::error::LLMError::ProviderError(e.to_string())),
        });

        Ok(Box::pin(stream))
    }
}

#[async_trait]
impl CompletionProvider for BedrockBackend {
    async fn complete(
        &self,
        req: &GenericCompletionRequest,
    ) -> std::result::Result<GenericCompletionResponse, crate::error::LLMError> {
        let request = CompletionRequest::new(&req.prompt);
        let response = self
            .complete_request(request)
            .await
            .map_err(|e| crate::error::LLMError::ProviderError(e.to_string()))?;

        Ok(GenericCompletionResponse {
            text: response.text,
        })
    }
}

#[async_trait]
impl EmbeddingProvider for BedrockBackend {
    async fn embed(
        &self,
        inputs: Vec<String>,
    ) -> std::result::Result<Vec<Vec<f32>>, crate::error::LLMError> {
        let mut embeddings = Vec::new();
        for input in inputs {
            let request = EmbeddingRequest::new(input);
            let response = self
                .embed_request(request)
                .await
                .map_err(|e| crate::error::LLMError::ProviderError(e.to_string()))?;
            let embedding_f32: Vec<f32> = response.embedding.iter().map(|&x| x as f32).collect();
            embeddings.push(embedding_f32);
        }
        Ok(embeddings)
    }
}

impl LLMProvider for BedrockBackend {}

/// Bedrock body for a Cohere embed call over `inputs` (1..=96 texts).
fn embed_batch_body(inputs: Vec<String>, input_type: Option<String>) -> Result<Value> {
    if inputs.is_empty() || inputs.len() > 96 {
        return Err(BedrockError::InvalidRequest(
            "cohere embed takes 1..=96 texts".to_string(),
        ));
    }
    Ok(json!({
        "texts": inputs,
        "input_type": input_type.unwrap_or_else(|| "search_document".to_string()),
        "embedding_types": ["float"],
    }))
}

/// Converts chat history to Converse messages with native `toolUse`/`toolResult` parts.
/// Messages are merged with a same-role neighbour only when tool parts are involved, since
/// Converse needs alternating roles and all results of one turn in a single user message.
fn history_to_messages(messages: &[LlmChatMessage]) -> Vec<ChatMessage> {
    let has_tool_part = |parts: &[ContentPart]| {
        parts.iter().any(|p| {
            matches!(
                p,
                ContentPart::ToolUse { .. } | ContentPart::ToolResult { .. }
            )
        })
    };

    let mut merged: Vec<(&'static str, Vec<ContentPart>)> = Vec::new();
    for m in messages {
        let role = match m.role {
            crate::chat::ChatRole::User => "user",
            crate::chat::ChatRole::Assistant => "assistant",
        };
        let parts = history_parts(m);
        if parts.is_empty() {
            continue;
        }
        match merged.last_mut() {
            Some((last_role, last_parts))
                if *last_role == role && (has_tool_part(last_parts) || has_tool_part(&parts)) =>
            {
                last_parts.extend(parts)
            }
            _ => merged.push((role, parts)),
        }
    }

    merged
        .into_iter()
        .map(|(role, mut parts)| {
            // Results lead a user message; tool calls trail assistant text.
            parts.sort_by_key(|p| match p {
                ContentPart::ToolResult { .. } => 0,
                ContentPart::ToolUse { .. } => 2,
                _ => 1,
            });
            let content = match parts.as_slice() {
                [ContentPart::Text { text }] => MessageContent::Text(text.clone()),
                _ => MessageContent::MultiModal(parts),
            };
            ChatMessage {
                role: role.to_string(),
                content,
            }
        })
        .collect()
}

fn history_parts(m: &LlmChatMessage) -> Vec<ContentPart> {
    let text = || ContentPart::Text {
        text: m.content.clone(),
    };
    match &m.message_type {
        crate::chat::MessageType::Image((mime, bytes)) => vec![
            text(),
            ContentPart::Image {
                source: bytes.clone(),
                media_type: mime.mime_type().to_string(),
            },
        ],
        crate::chat::MessageType::ToolUse(calls) => {
            let leading_text = (!m.content.trim().is_empty()).then(text);
            leading_text
                .into_iter()
                .chain(calls.iter().map(|call| ContentPart::ToolUse {
                    id: call.id.clone(),
                    name: call.function.name.clone(),
                    input: tool_input(&call.function.arguments),
                }))
                .collect()
        }
        crate::chat::MessageType::ToolResult(results) => results
            .iter()
            .map(|result| ContentPart::ToolResult {
                tool_use_id: result.id.clone(),
                // Converse rejects blank text blocks.
                content: if result.function.arguments.trim().is_empty() {
                    "[empty]".to_string()
                } else {
                    result.function.arguments.clone()
                },
                is_error: false,
            })
            .collect(),
        _ => vec![text()],
    }
}

/// Tool input must be a JSON object; unparsable or non-object arguments are kept
/// verbatim under `raw_arguments` rather than dropping the call.
fn tool_input(arguments: &str) -> Value {
    match serde_json::from_str::<Value>(arguments) {
        Ok(value @ Value::Object(_)) => value,
        _ if arguments.trim().is_empty() => json!({}),
        _ => json!({ "raw_arguments": arguments }),
    }
}

const PLACEHOLDER_TOOL_NAME: &str = "internal_placeholder_tool";

/// Converse rejects toolUse/toolResult blocks without a toolConfig. When history has them but
/// no real tools are offered, declare an uncallable placeholder so the request is valid.
/// `enforce_none` omits the tool choice so the caller can disable tools via the model's own field.
fn placeholder_tool_config(enforce_none: bool) -> Result<ToolConfiguration> {
    let invalid = |e: &dyn std::fmt::Debug| {
        BedrockError::InvalidRequest(format!("Failed to build placeholder tool: {:?}", e))
    };
    let spec = aws_sdk_bedrockruntime::types::ToolSpecification::builder()
        .name(PLACEHOLDER_TOOL_NAME)
        .description("Unavailable. Never call this tool.")
        .input_schema(ToolInputSchema::Json(BedrockBackend::value_to_document(
            &json!({"type": "object", "properties": {}}),
        )))
        .build()
        .map_err(|e| invalid(&e))?;
    // Bedrock rejects a toolChoice next to an additional `tool_choice` field.
    let tool_choice = (!enforce_none).then(|| {
        aws_sdk_bedrockruntime::types::ToolChoice::Auto(
            aws_sdk_bedrockruntime::types::AutoToolChoice::builder().build(),
        )
    });
    ToolConfiguration::builder()
        .tools(Tool::ToolSpec(spec))
        .set_tool_choice(tool_choice)
        .build()
        .map_err(|e| invalid(&e))
}

fn has_tool_blocks(messages: &[Message]) -> bool {
    messages.iter().flat_map(|m| m.content()).any(|block| {
        matches!(
            block,
            ContentBlock::ToolUse(_) | ContentBlock::ToolResult(_)
        )
    })
}

/// Converse has no "none" tool choice; Anthropic models accept their native one as an
/// additional request field. Other model families cannot disable tools with tool history.
fn is_anthropic_model(model: &BedrockModel) -> bool {
    model.model_id().contains("anthropic.")
}

/// Bedrock body for a Cohere rerank call over `documents` (1..=1000).
fn rerank_body(query: String, documents: Vec<String>, top_n: Option<usize>) -> Result<Value> {
    if documents.is_empty() || documents.len() > 1000 {
        return Err(BedrockError::InvalidRequest(
            "cohere rerank takes 1..=1000 documents".to_string(),
        ));
    }
    let top_n = top_n.unwrap_or(documents.len());
    Ok(json!({
        "query": query,
        "documents": documents,
        "top_n": top_n,
        "api_version": 2,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_backend_creation() {
        // This test requires AWS credentials
        // Run with: AWS_PROFILE=your-profile cargo test
        let result = BedrockBackend::from_env().await;
        assert!(result.is_ok() || matches!(result, Err(BedrockError::ConfigurationError(_))));
    }

    #[test]
    fn test_model_capability_overrides_toml_array() {
        let toml = r#"
[[model]]
name = "arn:aws:bedrock:eu-central-1:876164100382:inference-profile/eu.anthropic.claude-sonnet-4-20250514-v1:0"
completion = true
chat = true
embeddings = false
vision = true
tool_use = true
streaming = true
"#;

        let overrides = BedrockBackend::parse_model_capability_overrides(toml)
            .expect("TOML overrides should parse");
        let model = BedrockModel::from_id(
            "arn:aws:bedrock:eu-central-1:876164100382:inference-profile/eu.anthropic.claude-sonnet-4-20250514-v1:0",
        );

        assert_eq!(
            overrides.supports(&model, ModelCapability::Completion),
            Some(true)
        );
        assert_eq!(
            overrides.supports(&model, ModelCapability::Chat),
            Some(true)
        );
        assert_eq!(
            overrides.supports(&model, ModelCapability::Embeddings),
            Some(false)
        );
        assert_eq!(
            overrides.supports(&model, ModelCapability::Vision),
            Some(true)
        );
        assert_eq!(
            overrides.supports(&model, ModelCapability::ToolUse),
            Some(true)
        );
        assert_eq!(
            overrides.supports(&model, ModelCapability::Streaming),
            Some(true)
        );
    }

    #[test]
    fn test_prepare_chat_request_no_cache_point_without_cache_control() {
        let backend = BedrockBackend::new(
            "us-east-1".to_string(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let tools = vec![ToolDefinition {
            name: "get_weather".to_string(),
            description: "Get weather".to_string(),
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
            cache_control: None,
        }];

        let request = ChatRequest::new(vec![ChatMessage::user("hello")]).with_tools(tools);
        let prepared = backend.prepare_chat_request(request).unwrap();

        let tool_config = prepared.tool_config.expect("tool_config should be present");
        let tools = tool_config.tools();
        // Should only have the ToolSpec, no CachePoint
        assert_eq!(tools.len(), 1);
        assert!(matches!(tools[0], Tool::ToolSpec(_)));
    }

    #[test]
    fn test_prepare_chat_request_appends_cache_point_with_cache_control() {
        let backend = BedrockBackend::new(
            "us-east-1".to_string(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let tools = vec![ToolDefinition {
            name: "get_weather".to_string(),
            description: "Get weather".to_string(),
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
            cache_control: Some(serde_json::json!({"type": "ephemeral"})),
        }];

        let request = ChatRequest::new(vec![ChatMessage::user("hello")]).with_tools(tools);
        let prepared = backend.prepare_chat_request(request).unwrap();

        let tool_config = prepared.tool_config.expect("tool_config should be present");
        let tools = tool_config.tools();
        // Should have ToolSpec + CachePoint
        assert_eq!(tools.len(), 2);
        assert!(matches!(tools[0], Tool::ToolSpec(_)));
        assert!(matches!(tools[1], Tool::CachePoint(_)));
    }

    #[test]
    fn test_prepare_chat_request_cache_point_from_self_tools() {
        let llm_tools = vec![LlmTool {
            tool_type: "function".to_string(),
            function: crate::chat::FunctionTool {
                name: "search".to_string(),
                description: "Search".to_string(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
            },
            cache_control: Some(serde_json::json!({"type": "ephemeral"})),
        }];

        let backend = BedrockBackend::new(
            "us-east-1".to_string(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(llm_tools),
            None,
            None,
            None,
        )
        .unwrap();

        let request = ChatRequest::new(vec![ChatMessage::user("hello")]);
        let prepared = backend.prepare_chat_request(request).unwrap();

        let tool_config = prepared.tool_config.expect("tool_config should be present");
        let tools = tool_config.tools();
        // Should have ToolSpec + CachePoint
        assert_eq!(tools.len(), 2);
        assert!(matches!(tools[0], Tool::ToolSpec(_)));
        assert!(matches!(tools[1], Tool::CachePoint(_)));
    }

    #[test]
    fn a_max_tokens_stop_with_no_text_reports_its_reason_then_the_usage() {
        use aws_sdk_bedrockruntime::types::{
            ConverseStreamMetadataEvent, MessageStopEvent, StopReason, TokenUsage,
        };
        let mut tool_states = HashMap::new();
        let mut pending = VecDeque::new();
        let stop = MessageStopEvent::builder()
            .stop_reason(StopReason::MaxTokens)
            .build()
            .unwrap();
        converse_events(
            ConverseStreamOutput::MessageStop(stop),
            &mut tool_states,
            &mut pending,
        );
        let usage = TokenUsage::builder()
            .input_tokens(100)
            .output_tokens(1024)
            .total_tokens(1124)
            .build()
            .unwrap();
        let metadata = ConverseStreamMetadataEvent::builder().usage(usage).build();
        converse_events(
            ConverseStreamOutput::Metadata(metadata),
            &mut tool_states,
            &mut pending,
        );

        let events: Vec<ConverseEvent> = pending.into_iter().collect();
        assert_eq!(events.len(), 2, "{events:?}");
        assert!(
            matches!(
                &events[0],
                ConverseEvent::Chunk(LlmStreamChunk::Done { stop_reason }) if stop_reason == "max_tokens"
            ),
            "{events:?}"
        );
        match &events[1] {
            ConverseEvent::Usage(usage) => {
                assert_eq!(usage.input_tokens, 100);
                assert_eq!(usage.output_tokens, 1024);
                assert_eq!(usage.total_tokens, 1124);
            }
            other => panic!("expected the usage, got {other:?}"),
        }
    }

    #[test]
    fn the_struct_stream_carries_the_usage_as_a_final_chunk_without_choices() {
        let response = struct_response(ConverseEvent::Usage(UsageInfo {
            input_tokens: 100,
            output_tokens: 1024,
            total_tokens: 1124,
        }))
        .expect("the usage is a chunk");
        assert!(response.choices.is_empty());
        let usage = response.usage.expect("the usage is on the chunk");
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 1024);
        assert_eq!(usage.total_tokens, 1124);
    }

    #[test]
    fn the_struct_stream_names_the_stop_in_openai_words_on_an_empty_choice() {
        let response = struct_response(ConverseEvent::Chunk(LlmStreamChunk::Done {
            stop_reason: "max_tokens".into(),
        }))
        .expect("a stop is a chunk");
        assert_eq!(response.choices.len(), 1);
        let choice = &response.choices[0];
        assert!(choice.delta.content.is_none() && choice.delta.tool_calls.is_none());
        assert_eq!(choice.finish_reason.as_deref(), Some("length"));
        assert!(response.usage.is_none());
        assert_eq!(finish_reason_of("end_turn"), "stop");
        assert_eq!(finish_reason_of("tool_use"), "tool_calls");
        assert_eq!(finish_reason_of("guardrail_intervened"), "content_filter");
        assert_eq!(finish_reason_of("something_new"), "something_new");
    }

    #[test]
    fn test_tool_use_state_defaults_empty_arguments() {
        let state = BedrockToolUseState {
            id: "tooluse_1".to_string(),
            name: "get_servers".to_string(),
            ..Default::default()
        };

        let tool_call = state.to_tool_call();

        assert_eq!(tool_call.function.arguments, "{}");
    }

    fn call(id: &str, name: &str, arguments: &str) -> ToolCall {
        ToolCall {
            id: id.to_string(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: name.to_string(),
                arguments: arguments.to_string(),
            },
        }
    }

    fn parts(message: &ChatMessage) -> &[ContentPart] {
        match &message.content {
            MessageContent::MultiModal(parts) => parts,
            MessageContent::Text(_) => panic!("expected multimodal content"),
        }
    }

    #[test]
    fn test_history_sends_tool_call_and_result_natively() {
        let search = call("call_1", "search_tools", r#"{"query":"pull requests"}"#);
        let result = call(
            "call_1",
            "search_tools",
            r#"{"added_tools":["list_pull_requests"]}"#,
        );
        let history = vec![
            LlmChatMessage::user()
                .content("Find open pull requests")
                .build(),
            LlmChatMessage::assistant().tool_use(vec![search]).build(),
            LlmChatMessage::user().tool_result(vec![result]).build(),
        ];

        let messages = history_to_messages(&history);

        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1].role, "assistant");
        match parts(&messages[1]) {
            [ContentPart::ToolUse { id, name, input }] => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "search_tools");
                assert_eq!(input, &json!({"query": "pull requests"}));
            }
            other => panic!("unexpected parts: {other:?}"),
        }
        assert_eq!(messages[2].role, "user");
        match parts(&messages[2]) {
            [ContentPart::ToolResult {
                tool_use_id,
                content,
                is_error: false,
            }] => {
                assert_eq!(tool_use_id, "call_1");
                assert!(content.contains("list_pull_requests"));
            }
            other => panic!("unexpected parts: {other:?}"),
        }
    }

    #[test]
    fn test_history_merges_parallel_results_and_orders_text_before_calls() {
        let history = vec![
            LlmChatMessage::user().content("go").build(),
            LlmChatMessage::assistant()
                .tool_use(vec![call("a", "one", "{}"), call("b", "two", "{}")])
                .build(),
            LlmChatMessage::assistant()
                .content("Checking both.")
                .build(),
            LlmChatMessage::user()
                .tool_result(vec![call("a", "one", "ok")])
                .build(),
            LlmChatMessage::user()
                .tool_result(vec![call("b", "two", "")])
                .build(),
        ];

        let messages = history_to_messages(&history);

        assert_eq!(messages.len(), 3);
        assert!(matches!(
            parts(&messages[1]),
            [
                ContentPart::Text { .. },
                ContentPart::ToolUse { .. },
                ContentPart::ToolUse { .. }
            ]
        ));
        match parts(&messages[2]) {
            [ContentPart::ToolResult {
                tool_use_id: first, ..
            }, ContentPart::ToolResult {
                tool_use_id: second,
                content,
                ..
            }] => {
                assert_eq!((first.as_str(), second.as_str()), ("a", "b"));
                assert_eq!(content, "[empty]");
            }
            other => panic!("unexpected parts: {other:?}"),
        }
    }

    #[test]
    fn test_history_leaves_plain_text_messages_unmerged() {
        let history = vec![
            LlmChatMessage::user().content("a").build(),
            LlmChatMessage::user().content("b").build(),
        ];

        assert_eq!(history_to_messages(&history).len(), 2);
    }

    #[test]
    fn test_tool_input_keeps_unparsable_arguments() {
        assert_eq!(tool_input(r#"{"a":1}"#), json!({"a": 1}));
        assert_eq!(tool_input(""), json!({}));
        assert_eq!(
            tool_input("{not json"),
            json!({"raw_arguments": "{not json"})
        );
    }

    fn backend_with_tool_choice(tool_choice: Option<LlmToolChoice>) -> BedrockBackend {
        BedrockBackend::new(
            "us-east-1".to_string(),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            tool_choice,
            None,
            None,
        )
        .unwrap()
    }

    fn llm_tool_history() -> Vec<LlmChatMessage> {
        vec![
            LlmChatMessage::user().content("go").build(),
            LlmChatMessage::assistant()
                .tool_use(vec![call("a", "get_weather", "{}")])
                .build(),
            LlmChatMessage::user()
                .tool_result(vec![call("a", "get_weather", "sunny")])
                .build(),
        ]
    }

    fn tool_history() -> Vec<ChatMessage> {
        history_to_messages(&llm_tool_history())
    }

    fn weather_tool() -> Vec<ToolDefinition> {
        vec![ToolDefinition {
            name: "get_weather".to_string(),
            description: "Get weather".to_string(),
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
            cache_control: None,
        }]
    }

    fn tool_names(config: &ToolConfiguration) -> Vec<String> {
        config
            .tools()
            .iter()
            .filter_map(|t| match t {
                Tool::ToolSpec(spec) => Some(spec.name().to_string()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn test_prepare_chat_request_tool_history_without_tools_gets_placeholder() {
        let request = ChatRequest::new(tool_history());
        let prepared = backend_with_tool_choice(None)
            .prepare_chat_request(request)
            .unwrap();

        let config = prepared.tool_config.expect("tool_config should be present");
        assert_eq!(tool_names(&config), vec![PLACEHOLDER_TOOL_NAME]);
        // Converse requires the input schema to declare `"type": "object"`.
        let Some(Tool::ToolSpec(spec)) = config.tools().first() else {
            panic!("expected a tool spec");
        };
        let Some(ToolInputSchema::Json(Document::Object(schema))) = spec.input_schema() else {
            panic!("expected a JSON object schema");
        };
        assert_eq!(
            schema.get("type"),
            Some(&Document::String("object".to_string()))
        );
    }

    #[test]
    fn test_prepare_chat_request_tool_choice_none_disables_tools_natively() {
        let request = ChatRequest::new(tool_history()).with_tools(weather_tool());
        let prepared = backend_with_tool_choice(Some(LlmToolChoice::None))
            .prepare_chat_request(request)
            .unwrap();

        let config = prepared.tool_config.expect("tool_config should be present");
        assert_eq!(tool_names(&config), vec![PLACEHOLDER_TOOL_NAME]);
        assert!(config.tool_choice().is_none());
        assert_eq!(
            prepared
                .additional_model_request_fields
                .map(|fields| BedrockBackend::document_to_value(&fields)),
            Some(json!({"tool_choice": {"type": "none"}}))
        );
    }

    #[test]
    fn test_prepare_chat_request_tool_choice_none_rejected_for_non_anthropic_models() {
        let request = ChatRequest::new(tool_history());
        let result = backend_with_tool_choice(Some(LlmToolChoice::None))
            .with_model(BedrockModel::Direct(DirectModel::NovaProV1))
            .prepare_chat_request(request);

        assert!(matches!(result, Err(BedrockError::UnsupportedOperation(_))));
    }

    fn backend_with_effort(effort: &str) -> BedrockBackend {
        BedrockBackend {
            reasoning_effort: Some(effort.to_string()),
            ..backend_with_tool_choice(None)
        }
    }

    fn effort_fields(prepared: PreparedChatRequest) -> Option<Value> {
        prepared
            .additional_model_request_fields
            .map(|fields| BedrockBackend::document_to_value(&fields))
    }

    #[test]
    fn test_prepare_chat_request_sends_effort_to_claude() {
        let request = ChatRequest::new(history_to_messages(&[LlmChatMessage::user()
            .content("go")
            .build()]));
        let prepared = backend_with_effort("low")
            .prepare_chat_request(request)
            .unwrap();

        assert_eq!(
            effort_fields(prepared),
            Some(json!({"output_config": {"effort": "low"}}))
        );
    }

    #[test]
    fn test_prepare_chat_request_keeps_effort_beside_tool_choice_none() {
        let backend = BedrockBackend {
            tool_choice: Some(LlmToolChoice::None),
            ..backend_with_effort("low")
        };
        let request = ChatRequest::new(tool_history()).with_tools(weather_tool());
        let prepared = backend.prepare_chat_request(request).unwrap();

        assert_eq!(
            effort_fields(prepared),
            Some(json!({
                "tool_choice": {"type": "none"},
                "output_config": {"effort": "low"}
            }))
        );
    }

    #[test]
    fn test_prepare_chat_request_sends_no_effort_to_other_models() {
        let request = ChatRequest::new(history_to_messages(&[LlmChatMessage::user()
            .content("go")
            .build()]));
        let prepared = backend_with_effort("low")
            .with_model(BedrockModel::Direct(DirectModel::NovaProV1))
            .prepare_chat_request(request)
            .unwrap();

        assert_eq!(effort_fields(prepared), None);
    }

    /// Backend whose client sends to `endpoint` (a local mock) with dummy credentials.
    fn backend_against(endpoint: &str, tool_choice: Option<LlmToolChoice>) -> BedrockBackend {
        let backend = backend_with_tool_choice(tool_choice);
        let config = aws_sdk_bedrockruntime::Config::builder()
            .behavior_version(BehaviorVersion::latest())
            .region(aws_sdk_bedrockruntime::config::Region::new("us-east-1"))
            .endpoint_url(endpoint)
            .credentials_provider(aws_credential_types::Credentials::new(
                "test", "test", None, None, "test",
            ))
            .build();
        assert!(backend.client.set(BedrockClient::from_conf(config)).is_ok());
        backend
    }

    /// True when the Converse body disables tools via Anthropic's own `tool_choice`.
    fn disables_tools_natively(request: &mockito::Request) -> bool {
        let body: Value = request
            .body()
            .ok()
            .and_then(|bytes| serde_json::from_slice(bytes).ok())
            .unwrap_or_default();
        body["additionalModelRequestFields"] == json!({"tool_choice": {"type": "none"}})
            && body["toolConfig"].get("toolChoice").is_none()
            && body["toolConfig"]["tools"][0]["toolSpec"]["name"] == PLACEHOLDER_TOOL_NAME
    }

    #[tokio::test]
    async fn test_chat_with_tools_sends_tool_choice_none_with_tool_history() {
        let mut server = mockito::Server::new_async().await;
        let converse = server
            .mock(
                "POST",
                mockito::Matcher::Regex(r"^/model/.+/converse$".into()),
            )
            .match_request(disables_tools_natively)
            .with_status(200)
            .with_header("content-type", "application/json")
            .with_body(
                json!({
                    "output": {"message": {"role": "assistant", "content": [{"text": "done"}]}},
                    "stopReason": "end_turn",
                    "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
                    "metrics": {"latencyMs": 1}
                })
                .to_string(),
            )
            .expect(1)
            .create_async()
            .await;
        let backend = backend_against(&server.url(), Some(LlmToolChoice::None));

        let response = backend
            .chat_with_tools(&llm_tool_history(), None)
            .await
            .expect("request disabling tools should be accepted");

        assert_eq!(response.text().as_deref(), Some("done"));
        converse.assert_async().await;
    }

    #[tokio::test]
    async fn test_chat_stream_with_tools_sends_tool_choice_none_with_tool_history() {
        let mut server = mockito::Server::new_async().await;
        // Only the request is under test; a 400 ends the stream without retries.
        let converse_stream = server
            .mock(
                "POST",
                mockito::Matcher::Regex(r"^/model/.+/converse-stream$".into()),
            )
            .match_request(disables_tools_natively)
            .with_status(400)
            .with_header("content-type", "application/json")
            .with_body(r#"{"message":"stop here"}"#)
            .expect(1)
            .create_async()
            .await;
        let backend = backend_against(&server.url(), Some(LlmToolChoice::None));

        if let Ok(mut stream) =
            ChatProvider::chat_stream_with_tools(&backend, &llm_tool_history(), None).await
        {
            while stream.next().await.is_some() {}
        }

        converse_stream.assert_async().await;
    }

    #[tokio::test]
    async fn test_entry_points_reject_tool_choice_none_for_non_anthropic_models() {
        let mut server = mockito::Server::new_async().await;
        let any_request = server
            .mock("POST", mockito::Matcher::Any)
            .expect(0)
            .create_async()
            .await;
        let backend = backend_against(&server.url(), Some(LlmToolChoice::None))
            .with_model(BedrockModel::Direct(DirectModel::NovaProV1));

        let chat = backend.chat_with_tools(&llm_tool_history(), None).await;
        let stream =
            ChatProvider::chat_stream_with_tools(&backend, &llm_tool_history(), None).await;

        assert!(chat
            .err()
            .is_some_and(|e| e.to_string().contains("cannot disable tools")));
        assert!(stream
            .err()
            .is_some_and(|e| e.to_string().contains("cannot disable tools")));
        any_request.assert_async().await;
    }

    #[test]
    fn test_prepare_chat_request_tool_history_with_auto_keeps_real_tools_only() {
        let request = ChatRequest::new(tool_history()).with_tools(weather_tool());
        let prepared = backend_with_tool_choice(Some(LlmToolChoice::Auto))
            .prepare_chat_request(request)
            .unwrap();

        let config = prepared.tool_config.expect("tool_config should be present");
        assert_eq!(tool_names(&config), vec!["get_weather"]);
    }

    #[test]
    fn test_prepare_chat_request_text_history_without_tools_has_no_tool_config() {
        let request = ChatRequest::new(vec![ChatMessage::user("hello")]);
        let prepared = backend_with_tool_choice(None)
            .prepare_chat_request(request)
            .unwrap();

        assert!(prepared.tool_config.is_none());
    }

    #[test]
    fn test_prepare_chat_request_nova_uses_output_config_not_tool() {
        // Nova models must use outputConfig.textFormat for structured output,
        // not the synthetic json_schema_tool workaround.
        let backend = BedrockBackend::new(
            "us-east-1".to_string(),
            Some("amazon.nova-pro-v1:0".to_string()),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(crate::chat::StructuredOutputFormat {
                name: "result".to_string(),
                description: None,
                schema: Some(serde_json::json!({"type": "object", "properties": {}})),
                strict: None,
            }),
        )
        .unwrap();

        let request = ChatRequest::new(vec![ChatMessage::user("hello")]);
        let prepared = backend.prepare_chat_request(request).unwrap();

        // output_config must be set for the native path
        assert!(prepared.output_config.is_some());
        // tool_config must be None -- no synthetic json_schema_tool should be injected
        assert!(prepared.tool_config.is_none());
    }

    #[test]
    fn test_prepare_chat_request_claude_uses_tool_not_output_config() {
        // Claude (and other non-Nova models) must use the tool-based workaround.
        let backend = BedrockBackend::new(
            "us-east-1".to_string(),
            Some("us.anthropic.claude-sonnet-4-0-v1:0".to_string()),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(crate::chat::StructuredOutputFormat {
                name: "result".to_string(),
                description: None,
                schema: Some(serde_json::json!({"type": "object", "properties": {}})),
                strict: None,
            }),
        )
        .unwrap();

        let request = ChatRequest::new(vec![ChatMessage::user("hello")]);
        let prepared = backend.prepare_chat_request(request).unwrap();

        // tool_config must contain the synthetic json_schema_tool
        let tool_config = prepared
            .tool_config
            .expect("tool_config should be present for Claude");
        assert!(tool_config
            .tools()
            .iter()
            .any(|t| matches!(t, Tool::ToolSpec(spec) if spec.name() == "json_schema_tool")));
        // output_config must not be set
        assert!(prepared.output_config.is_none());
    }

    #[test]
    fn test_prepare_chat_request_nova_with_real_tools_and_schema() {
        // When Nova has both real user tools AND json_schema set, real tools go into
        // tool_config and the schema goes into output_config. Both can coexist because
        // the synthetic json_schema_tool is never injected on the native path.
        let backend = BedrockBackend::new(
            "us-east-1".to_string(),
            Some("amazon.nova-pro-v1:0".to_string()),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            Some(crate::chat::StructuredOutputFormat {
                name: "result".to_string(),
                description: None,
                schema: Some(serde_json::json!({"type": "object", "properties": {}})),
                strict: None,
            }),
        )
        .unwrap();

        let tools = vec![ToolDefinition {
            name: "get_weather".to_string(),
            description: "Get weather".to_string(),
            input_schema: serde_json::json!({"type": "object", "properties": {}}),
            cache_control: None,
        }];

        let request = ChatRequest::new(vec![ChatMessage::user("hello")]).with_tools(tools);
        let prepared = backend.prepare_chat_request(request).unwrap();

        assert!(prepared.output_config.is_some());
        let tool_config = prepared
            .tool_config
            .expect("real tools should produce tool_config");
        let tools = tool_config.tools();
        // Only the real tool, no json_schema_tool pollution
        assert_eq!(tools.len(), 1);
        assert!(matches!(tools[0], Tool::ToolSpec(ref spec) if spec.name() == "get_weather"));
    }

    #[test]
    fn test_embed_batch_body_sends_all_texts() {
        let body = embed_batch_body(vec!["a".into(), "b".into()], None).unwrap();
        assert_eq!(
            body,
            json!({"texts": ["a", "b"], "input_type": "search_document", "embedding_types": ["float"]})
        );
    }

    #[test]
    fn test_embed_batch_body_rejects_empty_and_over_96() {
        assert!(embed_batch_body(vec![], None).is_err());
        assert!(embed_batch_body(vec!["x".into(); 97], None).is_err());
        assert!(embed_batch_body(vec!["x".into(); 96], None).is_ok());
    }

    #[test]
    fn test_rerank_body_defaults_top_n_to_document_count() {
        let body = rerank_body("q".into(), vec!["a".into(), "b".into()], None).unwrap();
        assert_eq!(
            body,
            json!({"query": "q", "documents": ["a", "b"], "top_n": 2, "api_version": 2})
        );
    }

    #[test]
    fn test_rerank_body_rejects_empty_and_over_1000() {
        assert!(rerank_body("q".into(), vec![], None).is_err());
        assert!(rerank_body("q".into(), vec!["x".into(); 1001], None).is_err());
    }
}
