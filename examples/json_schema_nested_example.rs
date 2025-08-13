/// Example demonstrating complex nested JSON schema handling with LLM function calls
///
/// This example shows how to:
/// - Define a complex JSON schema for event creation with nested data structures
/// - Process chat messages with function calls
/// - Handle nested JSON responses
/// - Manage a multi-turn conversation with tool results
use llm::{
    builder::{FunctionBuilder, LLMBackend, LLMBuilder},
    chat::ChatMessage,
    FunctionCall, ToolCall,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let api_key = std::env::var("GOOGLE_API_KEY").unwrap_or("test-key".to_string());

    let llm = LLMBuilder::new()
        .backend(LLMBackend::Google)
        .api_key(api_key)
        .model("gemini-2.0-flash")
        .function(
            FunctionBuilder::new("create_event")
                .description("Creates a complex event with deeply nested data structures")
                .json_schema(json!({
                    "type": "object",
                    "properties": {
                        "event": {
                            "type": "object",
                            "properties": {
                                "title": {"type": "string"},
                                "location": {
                                    "type": "object",
                                    "properties": {
                                        "venue": {"type": "string"},
                                        "address": {
                                            "type": "object",
                                            "properties": {
                                                "street": {"type": "string"},
                                                "city": {"type": "string"},
                                                "coordinates": {
                                                    "type": "object",
                                                    "properties": {
                                                        "lat": {"type": "number"},
                                                        "lng": {"type": "number"}
                                                    }
                                                }
                                            }
                                        }
                                    }
                                },
                                "attendees": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "email": {"type": "string"},
                                            "role": {"type": "string"},
                                            "preferences": {
                                                "type": "object",
                                                "properties": {
                                                    "dietary": {"type": "string"},
                                                    "accessibility": {"type": "boolean"},
                                                    "notifications": {
                                                        "type": "object",
                                                        "properties": {
                                                            "email": {"type": "boolean"},
                                                            "sms": {"type": "boolean"},
                                                            "schedule": {
                                                                "type": "array",
                                                                "items": {
                                                                    "type": "object",
                                                                    "properties": {
                                                                        "time": {"type": "string"},
                                                                        "type": {"type": "string"}
                                                                    }
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                    "required": ["event"]
                })),
        )
        .build()?;

    let messages = vec![ChatMessage::user()
        .content("Create a team meeting at Google HQ in Mountain View for Alice (alice@corp.com, manager, vegetarian, needs accessibility, wants email and SMS notifications 1 hour before) and Bob (bob@corp.com, developer, no dietary restrictions, only email notifications 30 minutes before)")
        .build()];

    let response = llm.chat_with_tools(&messages, llm.tools()).await?;

    if let Some(tool_calls) = response.tool_calls() {
        println!("Complex nested schema handled successfully!");
        for call in &tool_calls {
            println!("Function: {}", call.function.name);
            let args: serde_json::Value = serde_json::from_str(&call.function.arguments)?;
            println!("Nested arguments: {}", serde_json::to_string_pretty(&args)?);

            let result = process_tool_call(call)?;
            println!("Result: {}", serde_json::to_string_pretty(&result)?);
        }

        let mut conversation = messages;
        conversation.push(
            ChatMessage::assistant()
                .tool_use(tool_calls.clone())
                .build(),
        );

        let tool_results: Vec<ToolCall> = tool_calls
            .iter()
            .map(|call| {
                let result = process_tool_call(call).unwrap();
                ToolCall {
                    id: call.id.clone(),
                    call_type: "function".to_string(),
                    function: FunctionCall {
                        name: call.function.name.clone(),
                        arguments: serde_json::to_string(&result).unwrap(),
                    },
                }
            })
            .collect();

        conversation.push(ChatMessage::user().tool_result(tool_results).build());

        let final_response = llm.chat_with_tools(&conversation, llm.tools()).await?;
        println!("\nFinal response: {final_response}");
    } else {
        println!("Direct response: {response}");
    }

    Ok(())
}

/// Processes a tool call and returns a simulated response
///
/// # Arguments
/// * `tool_call` - The tool call to process containing function name and arguments
///
/// # Returns
/// * JSON response containing event details or error message
fn process_tool_call(
    tool_call: &ToolCall,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    match tool_call.function.name.as_str() {
        "create_event" => {
            let args: serde_json::Value = serde_json::from_str(&tool_call.function.arguments)?;

            Ok(json!({
                "event_id": "evt_12345",
                "status": "created",
                "created_at": "2025-01-06T10:30:00Z",
                "calendar_links": {
                    "google": "https://calendar.google.com/event/evt_12345",
                    "outlook": "https://outlook.com/event/evt_12345"
                },
                "notifications_scheduled": args["event"]["attendees"]
                    .as_array()
                    .unwrap_or(&vec![])
                    .iter()
                    .map(|attendee| {
                        json!({
                            "attendee": attendee["email"],
                            "notifications": attendee["preferences"]["notifications"]
                        })
                    })
                    .collect::<Vec<_>>()
            }))
        }
        _ => Ok(json!({
            "error": "Unknown function",
            "function": tool_call.function.name
        })),
    }
}
