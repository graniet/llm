//! AWS Bedrock Tool Calling Example
//!
//! This example demonstrates how to use the AWS Bedrock backend with tool calling (function calling).
//!
//! To run this example:
//! cargo run --example bedrock_tool_calling_example --features bedrock

#[cfg(feature = "bedrock")]
use llm::backends::aws::*;
#[cfg(feature = "bedrock")]
use serde_json::json;

#[cfg(not(feature = "bedrock"))]
fn main() {
    println!("This example requires the 'bedrock' feature.");
    println!("Run with: cargo run --example bedrock_tool_calling_example --features bedrock");
}

#[cfg(feature = "bedrock")]
#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Initialize the backend
    let backend = match BedrockBackend::from_env().await {
        Ok(b) => b,
        Err(e) => {
            println!("Failed to initialize backend: {:?}", e);
            return Ok(());
        }
    };

    println!("Using region: {}", backend.region());

    // Define a tool
    let tools = vec![ToolDefinition {
        name: "get_weather".to_string(),
        description: "Get the current weather for a location".to_string(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit of temperature"
                }
            },
            "required": ["location"]
        }),
    }];

    // Start conversation
    let mut messages = vec![
        ChatMessage::user("What's the weather in London?"),
    ];

    println!("\nUser: What's the weather in London?");

    let request = ChatRequest::new(messages.clone())
        .with_model(BedrockModel::eu(CrossRegionModel::ClaudeSonnet4))
        .with_tools(tools.clone())
        .with_max_tokens(500);

    // First call: Model should request tool use
    let response = backend.chat_request(request).await?;
    
    // Add assistant's response to history
    messages.push(response.message.clone());

    // Check for tool use
    if let MessageContent::MultiModal(parts) = &response.message.content {
        for part in parts {
            if let ContentPart::ToolUse { id, name, input } = part {
                println!("Assistant wants to use tool '{}' with input: {}", name, input);
                
                if name == "get_weather" {
                    // Simulate tool execution
                    println!("Executing tool...");
                    let tool_result = json!({
                        "temperature": 15,
                        "unit": "celsius",
                        "condition": "cloudy"
                    }).to_string();

                    // Add tool result to history
                    messages.push(ChatMessage {
                        role: "user".to_string(),
                        content: MessageContent::MultiModal(vec![
                            ContentPart::ToolResult {
                                tool_use_id: id.clone(),
                                content: tool_result,
                                is_error: false,
                            }
                        ]),
                    });
                }
            }
        }
    } else {
        println!("Model did not use tools.");
        return Ok(());
    }

    // Second call: Model should interpret the result
    let request = ChatRequest::new(messages)
        .with_model(BedrockModel::eu(CrossRegionModel::ClaudeSonnet4))
        .with_tools(tools)
        .with_max_tokens(500);

    let response = backend.chat_request(request).await?;

    if let MessageContent::Text(text) = response.message.content {
        println!("Assistant: {}", text);
    } else if let MessageContent::MultiModal(parts) = response.message.content {
        for part in parts {
            if let ContentPart::Text { text } = part {
                println!("Assistant: {}", text);
            }
        }
    }

    Ok(())
}
