use axum::{http::StatusCode, Json};
use uuid::Uuid;

use super::helpers::{bad_request, internal_error, parse_model, transform_response, ApiResult};
use crate::api::types::{ChainStepRequest, ChatRequest, ChatResponse, Choice, Message};
use crate::api::ServerState;
use crate::chain::{MultiChainStep, MultiChainStepBuilder, MultiChainStepMode, MultiPromptChain};

const DEFAULT_TEMPERATURE: f32 = 0.7;
const DEFAULT_MAX_TOKENS: u32 = 1000;

pub async fn handle_chain_request(
    state: ServerState,
    req: ChatRequest,
) -> ApiResult<Json<ChatResponse>> {
    let last_step_id = resolve_last_step_id(&req)?;
    let mut provider_ids = Vec::new();
    let mut chain = MultiPromptChain::new(&state.llms);

    if let Some(model) = &req.model {
        chain = add_initial_step(chain, &mut provider_ids, &req, model)?;
    }

    let steps = build_steps(req.steps, &mut provider_ids)?;
    chain = chain.chain(steps);

    let chain_result = chain
        .run()
        .await
        .map_err(|e| internal_error(e.to_string()))?;

    let final_response = chain_result.get(&last_step_id).ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("No response found for step {last_step_id}"),
        )
    })?;

    Ok(Json(build_response(
        provider_ids.join(","),
        final_response.to_string(),
    )))
}

fn resolve_last_step_id(req: &ChatRequest) -> ApiResult<String> {
    if let Some(last_step) = req.steps.last() {
        return Ok(last_step.id.clone());
    }
    if req.model.is_some() {
        return Ok("initial".to_string());
    }
    Err(bad_request("No steps provided"))
}

fn add_initial_step<'a>(
    chain: MultiPromptChain<'a>,
    provider_ids: &mut Vec<String>,
    req: &ChatRequest,
    model: &str,
) -> ApiResult<MultiPromptChain<'a>> {
    let (provider_id, _) = parse_model(model)?;
    provider_ids.push(provider_id.clone());

    let prompt = last_message(req)
        .ok_or_else(|| bad_request("Initial model requires at least one message"))?;

    let transform = req.response_transform.clone().unwrap_or_default();
    let step = MultiChainStepBuilder::new(MultiChainStepMode::Chat)
        .provider_id(provider_id)
        .id("initial")
        .template(prompt)
        .max_tokens(req.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS))
        .temperature(req.temperature.unwrap_or(DEFAULT_TEMPERATURE))
        .response_transform(move |resp| transform_response(resp, &transform))
        .build()
        .map_err(|e| bad_request(e.to_string()))?;

    Ok(chain.step(step))
}

fn build_steps(
    steps: Vec<ChainStepRequest>,
    provider_ids: &mut Vec<String>,
) -> ApiResult<Vec<MultiChainStep>> {
    steps
        .into_iter()
        .map(|step| build_step(step, provider_ids))
        .collect()
}

fn build_step(step: ChainStepRequest, provider_ids: &mut Vec<String>) -> ApiResult<MultiChainStep> {
    provider_ids.push(step.provider_id.clone());
    let transform = step.response_transform.unwrap_or_default();

    MultiChainStepBuilder::new(MultiChainStepMode::Chat)
        .provider_id(step.provider_id)
        .id(step.id)
        .template(step.template)
        .temperature(step.temperature.unwrap_or(DEFAULT_TEMPERATURE))
        .max_tokens(step.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS))
        .response_transform(move |resp| transform_response(resp, &transform))
        .build()
        .map_err(|e| bad_request(e.to_string()))
}

fn last_message(req: &ChatRequest) -> Option<String> {
    req.messages
        .as_ref()
        .and_then(|messages| messages.last())
        .map(|message| message.content.clone())
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
