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
        ContentBlock, ContentBlockDelta, ConversationRole, ConverseStreamOutput, Message,
        SystemContentBlock, Tool, ToolConfiguration, ToolInputSchema, ToolResultBlock,
        ToolResultContentBlock, ToolUseBlock,
    },
    Client as BedrockClient,
};
use aws_smithy_types::{Blob, Document};
use futures::{Stream, StreamExt};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::OnceCell;

use crate::chat::{
    ChatMessage as LlmChatMessage, ChatProvider, StructuredOutputFormat, Tool as LlmTool,
    ToolChoice as LlmToolChoice,
};
use crate::completion::{
    CompletionProvider, CompletionRequest as GenericCompletionRequest,
    CompletionResponse as GenericCompletionResponse,
};
use crate::embedding::EmbeddingProvider;
use crate::models::ModelsProvider;
use crate::stt::SpeechToTextProvider;
use crate::tts::TextToSpeechProvider;
use crate::LLMProvider;

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
            model: model.map(BedrockModel::Custom),
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
                                let finish_reason = Some(format!("{:?}", stop.stop_reason));
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

        if let Some(tools) = request.tools {
            for tool in tools {
                bedrock_tools.push(self.convert_tool(&tool)?);
            }
        }

        let mut tool_choice = self.tool_choice.clone();
        if let Some(response_format) = self.json_schema.as_ref() {
            let schema = response_format.schema.clone().ok_or_else(|| {
                BedrockError::InvalidRequest(
                    "Structured output format must contain a schema".to_string(),
                )
            })?;

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

        if let Some(tools) = &self.tools {
            for tool in tools {
                bedrock_tools.push(self.convert_llm_tool(tool)?);
            }
        }

        let effective_tool_choice = tool_choice.unwrap_or(LlmToolChoice::Auto);
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
        let aws_messages: Vec<ChatMessage> = messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    crate::chat::ChatRole::User => "user",
                    crate::chat::ChatRole::Assistant => "assistant",
                };

                let content = match &m.message_type {
                    crate::chat::MessageType::Text => MessageContent::Text(m.content.clone()),
                    crate::chat::MessageType::Image((mime, bytes)) => {
                        MessageContent::MultiModal(vec![
                            ContentPart::Text {
                                text: m.content.clone(),
                            },
                            ContentPart::Image {
                                source: bytes.clone(),
                                media_type: mime.mime_type().to_string(),
                            },
                        ])
                    }
                    _ => MessageContent::Text(m.content.clone()),
                };

                ChatMessage {
                    role: role.to_string(),
                    content,
                }
            })
            .collect();

        let mut request = ChatRequest::new(aws_messages);

        if let Some(tools) = tools {
            let tool_defs: Vec<ToolDefinition> = tools
                .iter()
                .map(|t| ToolDefinition {
                    name: t.function.name.clone(),
                    description: t.function.description.clone(),
                    input_schema: t.function.parameters.clone(),
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
        let aws_messages: Vec<ChatMessage> = messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    crate::chat::ChatRole::User => "user",
                    crate::chat::ChatRole::Assistant => "assistant",
                };
                ChatMessage {
                    role: role.to_string(),
                    content: MessageContent::Text(m.content.clone()),
                }
            })
            .collect();

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
}
