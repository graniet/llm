//! Example demonstrating tool/function calling with Google's Gemini model
//!
//! This example shows how to:
//! - Configure a Google LLM with function calling capabilities
//! - Define a meeting scheduling function with JSON schema
//! - Process function calls and handle responses
//! - Maintain a conversation with tool usage

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
        .max_tokens(1024)
        .temperature(0.7)
        .function(
            FunctionBuilder::new("schedule_meeting")
                .description(
                    "Schedules a meeting with specified attendees at a given time and date.",
                )
                .json_schema(json!({
                    "type": "object",
                    "properties": {
                        "attendees": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of people attending the meeting."
                        },
                        "date": {
                            "type": "string",
                            "description": "Date of the meeting (e.g., '2024-07-29')"
                        },
                        "time": {
                            "type": "string",
                            "description": "Time of the meeting (e.g., '15:00')"
                        },
                        "topic": {
                            "type": "string",
                            "description": "The subject or topic of the meeting."
                        }
                    },
                    "required": ["attendees", "date", "time", "topic"]
                })),
        )
        .build()?;

    let messages = vec![ChatMessage::user()
        .content("Schedule a meeting with Bob and Alice for 03/27/2025 at 10:00 AM about the Q3 planning.")
        .build()];

    let response = llm.chat_with_tools(&messages, llm.tools()).await?;

    if let Some(tool_calls) = response.tool_calls() {
        println!("Tool calls requested:");
        for call in &tool_calls {
            println!("Function: {}", call.function.name);
            println!("Arguments: {}", call.function.arguments);

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

/// Processes a tool call by executing the requested function with provided arguments
///
/// # Arguments
/// * `tool_call` - The tool call containing function name and arguments to process
///
/// # Returns
/// * A JSON value containing the result of the function execution
///
/// # Errors
/// * If the function arguments cannot be parsed as JSON
/// * If an unknown function is called
fn process_tool_call(
    tool_call: &ToolCall,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    match tool_call.function.name.as_str() {
        "schedule_meeting" => {
            let args: serde_json::Value = serde_json::from_str(&tool_call.function.arguments)?;

            Ok(json!({
                "meeting_id": "mtg_12345",
                "status": "scheduled",
                "attendees": args["attendees"],
                "date": args["date"],
                "time": args["time"],
                "topic": args["topic"],
                "calendar_link": "https://calendar.google.com/event/mtg_12345"
            }))
        }
        _ => Ok(json!({
            "error": "Unknown function",
            "function": tool_call.function.name
        })),
    }
}
