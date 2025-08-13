use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    Json,
};
use uuid::Uuid;

use super::types::{ChatRequest, ChatResponse, Choice, Message};
use super::ServerState;
use crate::chat::{ChatMessage, ChatRole};
use crate::{
    chain::{MultiChainStepBuilder, MultiChainStepMode, MultiPromptChain},
    chat::MessageType,
};

/// Handles chat completion requests to the API server.
///
/// This handler processes incoming chat requests, validates authentication if required,
/// and routes the request to either a simple chat completion or a multi-step chain execution.
///
/// # Arguments
/// * `state` - Server state containing the LLM registry and authentication configuration
/// * `headers` - HTTP request headers containing authentication information
/// * `req` - Chat request payload containing messages, model specification, and optional chain steps
///
/// # Returns
/// * `Ok(Json<ChatResponse>)` - A successful chat completion response containing the generated message
/// * `Err((StatusCode, String))` - An error response with appropriate HTTP status code and message
///
/// # Authentication
/// If the server has an `auth_key` configured, this handler validates the Bearer token
/// in the Authorization header against the configured key.
///
/// # Request Processing
/// The handler supports two modes:
/// 1. Simple chat completion - Processes a single model request with messages
/// 2. Chain execution - Processes multiple steps through different models in sequence
///
/// # Model Specification
/// Models must be specified in the format "provider:model_name" (e.g. "openai:gpt-4", "anthropic:claude-2")
///
/// # Response Format
/// Returns a standardized chat completion response containing:
/// - Unique ID for the completion
/// - Unix timestamp of creation
/// - Model identifier
/// - Generated message content
/// - Choice metadata including finish reason
pub async fn handle_chat(
    State(state): State<ServerState>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, (StatusCode, String)> {
    if let Some(key) = &state.auth_key {
        let auth_header = headers.get("Authorization").ok_or((
            StatusCode::UNAUTHORIZED,
            "Missing authorization".to_string(),
        ))?;

        let auth_str = auth_header.to_str().map_err(|_| {
            (
                StatusCode::UNAUTHORIZED,
                "Invalid authorization header".to_string(),
            )
        })?;

        if !auth_str.starts_with("Bearer ") || &auth_str[7..] != key {
            return Err((StatusCode::UNAUTHORIZED, "Invalid API key".to_string()));
        }
    }

    if !req.steps.is_empty() {
        return handle_chain_request(state, req).await;
    }

    let messages: Vec<ChatMessage> = req
        .messages
        .unwrap_or(vec![])
        .into_iter()
        .map(|msg| ChatMessage {
            role: match msg.role.as_str() {
                "user" => ChatRole::User,
                "assistant" => ChatRole::Assistant,
                _ => ChatRole::User,
            },
            message_type: MessageType::Text,
            content: msg.content,
        })
        .collect();

    let (provider_id, model_name) = req
        .model
        .as_ref()
        .ok_or((StatusCode::BAD_REQUEST, "Model is required".to_string()))?
        .split_once(':')
        .ok_or((StatusCode::BAD_REQUEST, "Invalid model format".to_string()))?;

    let provider = state.llms.get(provider_id).ok_or((
        StatusCode::BAD_REQUEST,
        format!("Unknown provider: {provider_id}"),
    ))?;

    let response = provider
        .chat(&messages)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(ChatResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: model_name.to_string(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: response.text().unwrap_or_default(),
            },
            finish_reason: "stop".to_string(),
        }],
    }))
}

/// Handles multi-step chain requests by orchestrating message flow through multiple models.
///
/// This handler processes requests that specify a sequence of model interactions, where
/// each step can transform and pass its output to subsequent steps.
///
/// # Arguments
/// * `state` - Server state containing the LLM registry
/// * `req` - Chat request containing chain step specifications
///
/// # Returns
/// * `Ok(Json<ChatResponse>)` - Final chain output wrapped in a chat response
/// * `Err((StatusCode, String))` - Error response with status code and message
///
/// # Chain Processing
/// 1. Initializes chain with optional initial model
/// 2. Processes each step sequentially with specified transformations
/// 3. Returns final step's output in standardized format
async fn handle_chain_request(
    state: ServerState,
    req: ChatRequest,
) -> Result<Json<ChatResponse>, (StatusCode, String)> {
    let mut provider_ids = Vec::new();
    let mut chain = MultiPromptChain::new(&state.llms);

    let last_step_id = if let Some(last_step) = req.steps.last() {
        last_step.id.clone()
    } else if req.model.is_some() {
        "initial".to_string()
    } else {
        return Err((StatusCode::BAD_REQUEST, "No steps provided".to_string()));
    };

    let transform_response = |resp: String, transform: &str| -> String {
        match transform {
            "extract_think" => resp
                .lines()
                .skip_while(|line| !line.contains("<think>"))
                .take_while(|line| !line.contains("</think>"))
                .map(|line| line.replace("<think>", "").trim().to_string())
                .filter(|line| !line.is_empty())
                .collect::<Vec<_>>()
                .join("\n"),
            "trim_whitespace" => resp.trim().to_string(),
            "extract_json" => {
                let json_start = resp.find("```json").unwrap_or(0);
                let json_end = resp.find("```").unwrap_or(resp.len());
                let json_str = &resp[json_start..json_end];
                serde_json::from_str::<String>(json_str)
                    .unwrap_or_else(|_| "Invalid JSON response".to_string())
            }
            _ => resp.to_string(),
        }
    };

    if let Some(ref model) = req.model {
        let (provider_id, _) = model
            .split_once(':')
            .ok_or((StatusCode::BAD_REQUEST, "Invalid model format".to_string()))?;

        provider_ids.push(provider_id.to_string());
        let messages = req.messages.unwrap_or_default();

        chain = chain.step(
            MultiChainStepBuilder::new(MultiChainStepMode::Chat)
                .provider_id(provider_id.to_string())
                .id("initial")
                .template(messages.last().unwrap().content.clone())
                .max_tokens(req.max_tokens.unwrap_or(1000))
                .temperature(req.temperature.unwrap_or(0.7))
                .response_transform({
                    let transform = req.response_transform.unwrap_or_default();
                    move |resp| transform_response(resp, &transform)
                })
                .build()
                .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?,
        );
    }

    chain = chain.chain(
        req.steps
            .into_iter()
            .map(|step| {
                provider_ids.push(step.provider_id.clone());
                let transform = step.response_transform.unwrap_or_default();
                MultiChainStepBuilder::new(MultiChainStepMode::Chat)
                    .provider_id(step.provider_id)
                    .id(step.id)
                    .template(step.template)
                    .temperature(step.temperature.unwrap_or(0.7))
                    .max_tokens(step.max_tokens.unwrap_or(1000))
                    .response_transform(move |resp| transform_response(resp, &transform))
                    .build()
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?,
    );

    let chain_result = chain
        .run()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let final_response = chain_result.get(&last_step_id).ok_or((
        StatusCode::INTERNAL_SERVER_ERROR,
        format!("No response found for step {last_step_id}"),
    ))?;

    Ok(Json(ChatResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        model: provider_ids.join(",").to_string(),
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content: final_response.to_string(),
            },
            finish_reason: "stop".to_string(),
        }],
    }))
}
