use serde_json::Value;

use crate::chat::Usage;
use crate::error::LLMError;

const DONE_EVENT: &str = "[DONE]";

#[derive(Clone, Debug)]
pub(super) struct ToolState {
    pub(super) call_id: String,
    pub(super) name: String,
    pub(super) arguments: String,
    pub(super) output_index: usize,
}

pub(super) enum ResponsesEvent {
    OutputTextDelta {
        delta: String,
    },
    FunctionCallAdded {
        item_id: String,
        call_id: String,
        name: String,
        output_index: usize,
    },
    FunctionCallDelta {
        item_id: String,
        delta: String,
        output_index: usize,
    },
    FunctionCallDone {
        item_id: String,
        arguments: String,
        output_index: usize,
    },
    OutputItemDone {
        item_id: String,
        output_index: usize,
    },
    ResponseCompleted {
        usage: Option<Usage>,
    },
}

pub(super) fn extract_payload(buffer: &str) -> Option<String> {
    let mut payload = String::new();
    for line in buffer.lines() {
        if let Some(data) = line.strip_prefix("data: ") {
            if data == DONE_EVENT {
                return Some(DONE_EVENT.to_string());
            }
            payload.push_str(data);
        }
    }
    if payload.is_empty() {
        None
    } else {
        Some(payload)
    }
}

pub(super) fn parse_event(payload: &str) -> Result<Option<ResponsesEvent>, LLMError> {
    if payload == DONE_EVENT {
        return Ok(Some(ResponsesEvent::ResponseCompleted { usage: None }));
    }
    let value: Value = serde_json::from_str(payload)?;
    match event_type(&value) {
        Some("response.output_text.delta") => parse_output_text_delta(&value),
        Some("response.output_item.added") => parse_function_call_added(&value),
        Some("response.function_call_arguments.delta") => parse_function_call_delta(&value),
        Some("response.function_call_arguments.done") => parse_function_call_done(&value),
        Some("response.output_item.done") => parse_output_item_done(&value),
        Some("response.completed") => parse_response_completed(&value),
        _ => Ok(None),
    }
}

fn event_type(value: &Value) -> Option<&str> {
    value.get("type").and_then(|v| v.as_str())
}

fn parse_output_text_delta(value: &Value) -> Result<Option<ResponsesEvent>, LLMError> {
    let delta = value
        .get("delta")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    if delta.is_empty() {
        Ok(None)
    } else {
        Ok(Some(ResponsesEvent::OutputTextDelta { delta }))
    }
}

fn parse_function_call_added(value: &Value) -> Result<Option<ResponsesEvent>, LLMError> {
    let item = match value.get("item") {
        Some(item) => item,
        None => return Ok(None),
    };
    if item.get("type").and_then(|v| v.as_str()) != Some("function_call") {
        return Ok(None);
    }
    let item_id = string_field(item, "id")?;
    let call_id = string_field(item, "call_id")?;
    let name = string_field(item, "name")?;
    Ok(Some(ResponsesEvent::FunctionCallAdded {
        item_id,
        call_id,
        name,
        output_index: output_index(value),
    }))
}

fn parse_function_call_delta(value: &Value) -> Result<Option<ResponsesEvent>, LLMError> {
    let item_id = string_field(value, "item_id")?;
    let delta = string_field(value, "delta")?;
    Ok(Some(ResponsesEvent::FunctionCallDelta {
        item_id,
        delta,
        output_index: output_index(value),
    }))
}

fn parse_function_call_done(value: &Value) -> Result<Option<ResponsesEvent>, LLMError> {
    let item_id = string_field(value, "item_id")?;
    let arguments = string_field(value, "arguments")?;
    Ok(Some(ResponsesEvent::FunctionCallDone {
        item_id,
        arguments,
        output_index: output_index(value),
    }))
}

fn parse_output_item_done(value: &Value) -> Result<Option<ResponsesEvent>, LLMError> {
    let item = match value.get("item") {
        Some(item) => item,
        None => return Ok(None),
    };
    if item.get("type").and_then(|v| v.as_str()) != Some("function_call") {
        return Ok(None);
    }
    let item_id = string_field(item, "id")?;
    Ok(Some(ResponsesEvent::OutputItemDone {
        item_id,
        output_index: output_index(value),
    }))
}

fn parse_response_completed(value: &Value) -> Result<Option<ResponsesEvent>, LLMError> {
    let usage = value
        .get("response")
        .and_then(|resp| resp.get("usage"))
        .map(|usage| serde_json::from_value::<Usage>(usage.clone()))
        .transpose()?;
    Ok(Some(ResponsesEvent::ResponseCompleted { usage }))
}

fn string_field(value: &Value, key: &str) -> Result<String, LLMError> {
    value
        .get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .ok_or_else(|| LLMError::ResponseFormatError {
            message: format!("Missing {key} in responses event"),
            raw_response: value.to_string(),
        })
}

pub(super) fn output_index(value: &Value) -> usize {
    value
        .get("output_index")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize
}
