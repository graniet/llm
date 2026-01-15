use axum::{extract::State, http::HeaderMap, Json};
use uuid::Uuid;

use super::chain::handle_chain_request;
use super::helpers::{bad_request, internal_error, parse_model, unauthorized, ApiResult};
use crate::api::types::{ChatRequest, ChatResponse, Choice, Message};
use crate::api::ServerState;
use crate::chat::{ChatMessage, ChatRole, MessageType};

pub async fn handle_chat(
    State(state): State<ServerState>,
    headers: HeaderMap,
    Json(req): Json<ChatRequest>,
) -> ApiResult<Json<ChatResponse>> {
    validate_auth(&state, &headers)?;
    if !req.steps.is_empty() {
        return handle_chain_request(state, req).await;
    }

    let messages = build_messages(req.messages);
    let model = req.model.ok_or_else(|| bad_request("Model is required"))?;
    let (provider_id, model_name) = parse_model(&model)?;
    let provider = state
        .llms
        .get(&provider_id)
        .ok_or_else(|| bad_request(format!("Unknown provider: {provider_id}")))?;

    let response = provider
        .chat(&messages)
        .await
        .map_err(|e| internal_error(e.to_string()))?;

    Ok(Json(build_response(
        model_name,
        response.text().unwrap_or_default(),
    )))
}

fn validate_auth(state: &ServerState, headers: &HeaderMap) -> ApiResult<()> {
    let Some(key) = &state.auth_key else {
        return Ok(());
    };

    let auth_header = headers
        .get("Authorization")
        .ok_or_else(|| unauthorized("Missing authorization"))?;
    let auth_str = auth_header
        .to_str()
        .map_err(|_| unauthorized("Invalid authorization header"))?;

    if !auth_str.starts_with("Bearer ") || &auth_str[7..] != key {
        return Err(unauthorized("Invalid API key"));
    }

    Ok(())
}

fn build_messages(messages: Option<Vec<Message>>) -> Vec<ChatMessage> {
    messages
        .unwrap_or_default()
        .into_iter()
        .map(|msg| ChatMessage {
            role: parse_role(&msg.role),
            message_type: MessageType::Text,
            content: msg.content,
        })
        .collect()
}

fn parse_role(role: &str) -> ChatRole {
    match role {
        "user" => ChatRole::User,
        "assistant" => ChatRole::Assistant,
        _ => ChatRole::User,
    }
}

fn build_response(model: String, content: String) -> ChatResponse {
    ChatResponse {
        id: format!("chatcmpl-{}", Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        model,
        choices: vec![Choice {
            index: 0,
            message: Message {
                role: "assistant".to_string(),
                content,
            },
            finish_reason: "stop".to_string(),
        }],
    }
}
