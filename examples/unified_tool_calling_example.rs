// Unified Tool Calling Example - demonstrating LLM's abstraction across providers
//
// This example shows how to use tool calling in a provider-agnostic way.
// It works with any supported backend (OpenAI, Anthropic, Google) and
// demonstrates different tool calling scenarios.
//
// Usage:
//   cargo run --example unified_tool_calling_example -- [provider] [scenario]
//
// Providers: openai, anthropic, google
// Scenarios: simple, multi, choice

use llm::{
    builder::{FunctionBuilder, LLMBackend, LLMBuilder, ParamBuilder},
    chat::{ChatMessage, ToolChoice},
    FunctionCall, LLMProvider, ToolCall,
};
use serde_json::{json, Value};
use std::env;
use std::error::Error;

/// Main entry point
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Get command line arguments
    let args: Vec<String> = env::args().collect();

    // Default to OpenAI if no provider specified
    let provider = if args.len() > 1 { &args[1] } else { "openai" };

    // Default to simple scenario if not specified
    let scenario = if args.len() > 2 { &args[2] } else { "simple" };

    // Create an LLM instance with the specified provider
    let llm = create_llm(provider)?;

    // Print example information
    println!("=== Unified Tool Calling Example ===");
    println!("Provider: {provider}");
    println!("Scenario: {scenario}");
    println!("=================================\n");

    // Run the requested scenario
    match scenario {
        "simple" => run_simple_scenario(&llm).await?,
        "multi" => run_multi_turn_scenario(&llm).await?,
        "choice" => run_tool_choice_scenario(&llm).await?,
        _ => {
            println!("Unknown scenario: {scenario}. Available scenarios: simple, multi, choice");
            println!("Example: cargo run --example unified_tool_calling_example -- openai multi");
        }
    }

    Ok(())
}

/// Create an LLM instance with the specified provider and common tools
fn create_llm(provider_name: &str) -> Result<Box<dyn LLMProvider>, Box<dyn Error>> {
    // Parse the provider from string
    let backend = match provider_name.to_lowercase().as_str() {
        "openai" => LLMBackend::OpenAI,
        "anthropic" => LLMBackend::Anthropic,
        "google" => LLMBackend::Google,
        "ollama" => LLMBackend::Ollama,
        _ => {
            return Err(format!(
                "Unsupported provider: {provider_name}. Use 'openai', 'anthropic', or 'google'"
            )
            .into());
        }
    };

    // Get appropriate API key based on provider
    let api_key = match backend {
        LLMBackend::OpenAI => {
            env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set")
        }
        LLMBackend::Anthropic => {
            env::var("ANTHROPIC_API_KEY").expect("ANTHROPIC_API_KEY environment variable not set")
        }
        LLMBackend::Google => {
            env::var("GOOGLE_API_KEY").expect("GOOGLE_API_KEY environment variable not set")
        }
        LLMBackend::Ollama => env::var("OLLAMA_API_KEY").unwrap_or("ollama".into()),
        _ => unreachable!(),
    };

    // Get appropriate model name based on provider
    let model = match backend {
        LLMBackend::OpenAI => "gpt-4o-mini",
        LLMBackend::Anthropic => "claude-3-5-haiku-latest",
        LLMBackend::Google => "gemini-1.5-flash",
        LLMBackend::Ollama => "llama3.1:latest",
        _ => unreachable!(),
    };

    // Build the LLM with tools
    let llm = LLMBuilder::new()
        .backend(backend)
        .api_key(api_key)
        .model(model)
        .max_tokens(1024)
        .temperature(0.7)
        .function(
            FunctionBuilder::new("get_weather")
                .description("Get the current weather in a specific location")
                .param(ParamBuilder::new("location").type_of("string").description(
                    "The city and state/country, e.g., 'San Francisco, CA' or 'Tokyo, Japan'",
                ))
                .required(vec!["location".to_string()]),
        )
        .function(
            FunctionBuilder::new("get_current_time")
                .description("Get the current time in a specific time zone")
                .param(
                    ParamBuilder::new("timezone")
                        .type_of("string")
                        .description("The timezone, e.g., 'EST', 'PST', 'UTC', 'JST', etc."),
                )
                .required(vec!["timezone".to_string()]),
        )
        .function(
            FunctionBuilder::new("search_restaurants")
                .description("Search for restaurants in a specific location")
                .param(
                    ParamBuilder::new("location")
                        .type_of("string")
                        .description("The city or neighborhood to search in"),
                )
                .param(
                    ParamBuilder::new("cuisine")
                        .type_of("string")
                        .description("Type of cuisine, e.g., 'Italian', 'Japanese', etc."),
                )
                .required(vec!["location".to_string()]),
        )
        .build()?;

    Ok(llm)
}

/// Run a simple tool calling scenario - single query with tool use
async fn run_simple_scenario(llm: &Box<dyn LLMProvider>) -> Result<(), Box<dyn Error>> {
    println!("SCENARIO: Simple Tool Calling");
    println!("This demonstrates a single query that triggers tool use\n");

    // Create a query that should trigger tool use
    let messages = vec![ChatMessage::user()
        .content("What's the current weather in Tokyo?")
        .build()];

    println!("User: What's the current weather in Tokyo?\n");
    println!("Sending request to model...");

    // Send the request
    let response = llm.chat_with_tools(&messages, llm.tools()).await?;

    // Check if model wants to use tools
    if let Some(tool_calls) = response.tool_calls() {
        println!("\nModel is using tools: {}", tool_calls.len());

        for call in &tool_calls {
            println!("Tool call: {}", call.function.name);
            println!("Arguments: {}\n", call.function.arguments);

            // Process the tool call
            let result = process_tool_call(call)?;
            println!(
                "Tool response: {}\n",
                serde_json::to_string_pretty(&result)?
            );

            // Create a follow-up with tool results
            let mut follow_up = messages.clone();

            // Add the assistant's response with tool use
            follow_up.push(
                ChatMessage::assistant()
                    .tool_use(tool_calls.clone())
                    .content("")
                    .build(),
            );

            // Add the tool results
            let tool_results = vec![ToolCall {
                id: call.id.clone(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: call.function.name.clone(),
                    arguments: serde_json::to_string(&result)?,
                },
            }];

            follow_up.push(
                ChatMessage::user()
                    .tool_result(tool_results)
                    .content("")
                    .build(),
            );

            // Get final response
            println!("Getting final response with tool results...");
            let final_response = llm.chat_with_tools(&follow_up, llm.tools()).await?;
            println!("\nFinal response: {final_response}");
        }
    } else {
        println!("\nModel provided a direct response (no tools used):\n{response}");
    }

    Ok(())
}

/// Run a multi-turn conversation scenario
async fn run_multi_turn_scenario(llm: &Box<dyn LLMProvider>) -> Result<(), Box<dyn Error>> {
    println!("SCENARIO: Multi-turn Conversation with Tool Calling");
    println!("This demonstrates maintaining context across multiple turns with tool use\n");

    // Initialize conversation history
    let mut conversation = Vec::new();

    // First turn
    let user_query = "I'm planning a trip to Tokyo. What's the weather like there?";
    println!("User: {user_query}\n");

    await_tool_response(llm, &mut conversation, user_query).await?;

    // Second turn - building on previous context
    let user_query = "What time is it there right now?";
    println!("\nUser: {user_query}\n");

    await_tool_response(llm, &mut conversation, user_query).await?;

    // Third turn - building on full conversation context
    let user_query = "Can you recommend some good sushi restaurants in Tokyo?";
    println!("\nUser: {user_query}\n");

    await_tool_response(llm, &mut conversation, user_query).await?;

    // Fourth turn - follow-up that requires memory of entire conversation
    let user_query =
        "Based on the weather and time, when would be a good time to visit those restaurants?";
    println!("\nUser: {user_query}\n");

    await_tool_response(llm, &mut conversation, user_query).await?;

    Ok(())
}

/// Run a tool choice demonstration scenario
async fn run_tool_choice_scenario(llm: &Box<dyn LLMProvider>) -> Result<(), Box<dyn Error>> {
    println!("SCENARIO: Tool Choice Options");
    println!("This demonstrates controlling how the model uses tools\n");

    // Create a query that could use weather or time tools
    let query = "What's the weather like in Tokyo and what time is it there?";

    // Test Auto tool choice (default behavior)
    test_tool_choice(llm, ToolChoice::Auto, query).await?;

    // Test Any tool choice (model must use at least one tool)
    test_tool_choice(llm, ToolChoice::Any, query).await?;

    // Test specific tool choice (force weather tool)
    test_tool_choice(llm, ToolChoice::Tool("get_weather".to_string()), query).await?;

    // Test None tool choice (disable tools)
    test_tool_choice(llm, ToolChoice::None, query).await?;

    Ok(())
}

/// Helper function to test different tool choice settings
async fn test_tool_choice(
    llm: &Box<dyn LLMProvider>,
    tool_choice: ToolChoice,
    query: &str,
) -> Result<(), Box<dyn Error>> {
    println!("\n--- Testing {tool_choice:?} ---");

    // Create a custom LLM with the specified tool choice
    let mut builder = LLMBuilder::new();

    // Copy properties from the original LLM
    if let Some(tools) = llm.tools() {
        for tool in tools {
            builder = builder.function(
                FunctionBuilder::new(&tool.function.name).description(&tool.function.description), // Note: we'd need to recreate all parameters here in a real implementation
            );
        }
    }

    let custom_llm = builder
        .backend(match llm {
            _ if std::any::type_name::<OpenAI>().contains("OpenAI") => LLMBackend::OpenAI,
            _ if std::any::type_name::<Anthropic>().contains("Anthropic") => LLMBackend::Anthropic,
            _ if std::any::type_name::<Google>().contains("Google") => LLMBackend::Google,
            _ => LLMBackend::OpenAI, // Default fallback
        })
        .api_key(std::env::var("OPENAI_API_KEY").unwrap_or("key".to_string()))
        .model("gpt-4")
        .max_tokens(1024)
        .tool_choice(tool_choice.clone())
        .build()?;

    // Send query with the specified tool choice
    let messages = vec![ChatMessage::user().content(query).build()];

    println!("User: {query}\n");

    let response = custom_llm
        .chat_with_tools(&messages, custom_llm.tools())
        .await?;

    // Check tool usage
    if let Some(tool_calls) = response.tool_calls() {
        println!("Tools called:");
        for call in tool_calls {
            println!("- {}", call.function.name);
        }
    } else {
        println!("No tools called");
    }

    println!("\nResponse: {response}");

    Ok(())
}

/// Helper function for multi-turn conversation with tool handling
async fn await_tool_response(
    llm: &Box<dyn LLMProvider>,
    conversation: &mut Vec<ChatMessage>,
    user_query: &str,
) -> Result<(), Box<dyn Error>> {
    // Add user query to conversation
    conversation.push(ChatMessage::user().content(user_query).build());

    // Get model's response
    let response = llm.chat_with_tools(conversation, llm.tools()).await?;

    // Check if model wants to use tools
    if let Some(tool_calls) = response.tool_calls() {
        println!("Model is using tools: {}", tool_calls.len());

        // Add assistant's tool use message
        conversation.push(
            ChatMessage::assistant()
                .tool_use(tool_calls.clone())
                .content(response.text().unwrap_or_default())
                .build(),
        );

        // Process each tool call
        let mut tool_results = Vec::new();

        for call in &tool_calls {
            println!("Tool call: {}", call.function.name);
            println!("Arguments: {}", call.function.arguments);

            // Process the tool call
            let result = process_tool_call(call)?;
            println!("Tool response: {}", serde_json::to_string_pretty(&result)?);

            // Add to tool results
            tool_results.push(ToolCall {
                id: call.id.clone(),
                call_type: "function".to_string(),
                function: FunctionCall {
                    name: call.function.name.clone(),
                    arguments: serde_json::to_string(&result)?,
                },
            });
        }

        // Add tool results to conversation
        conversation.push(ChatMessage::user().tool_result(tool_results).build());

        // Get final response
        println!("Getting final response with tool results...");
        let final_response = llm.chat_with_tools(conversation, llm.tools()).await?;
        println!("\nAssistant: {}", final_response.text().unwrap_or_default());

        // Add assistant's response to conversation
        conversation.push(
            ChatMessage::assistant()
                .content(final_response.text().unwrap_or_default())
                .build(),
        );
    } else {
        // Direct response (no tools)
        let response_text = response.text().unwrap_or_default();

        println!("\nAssistant: {response_text}");

        // Add to conversation history
        conversation.push(ChatMessage::assistant().content(response_text).build());
    }

    Ok(())
}

/// Process a tool call and return a simulated response
fn process_tool_call(tool_call: &ToolCall) -> Result<Value, Box<dyn Error>> {
    // Parse the arguments as JSON
    let args: Value = serde_json::from_str(&tool_call.function.arguments)?;

    // Generate a response based on the tool type
    match tool_call.function.name.as_str() {
        "get_weather" => {
            let location = args["location"].as_str().unwrap_or("unknown location");

            Ok(json!({
                "location": location,
                "temperature": 22,
                "units": "celsius",
                "conditions": "Partly cloudy",
                "humidity": "65%",
                "forecast": "Clear skies expected later today"
            }))
        }
        "get_current_time" => {
            let timezone = args["timezone"].as_str().unwrap_or("UTC");

            Ok(json!({
                "timezone": timezone,
                "current_time": "14:30",
                "date": "April 2, 2025",
                "day_of_week": "Wednesday"
            }))
        }
        "search_restaurants" => {
            let location = args["location"].as_str().unwrap_or("unknown location");
            let cuisine = args["cuisine"].as_str().unwrap_or("any");

            Ok(json!({
                "location": location,
                "cuisine": cuisine,
                "restaurants": [
                    {
                        "name": "Sushi Dai",
                        "rating": 4.8,
                        "price_range": "$$$",
                        "specialty": "Omakase"
                    },
                    {
                        "name": "Tsukiji Sushisay",
                        "rating": 4.6,
                        "price_range": "$$",
                        "specialty": "Market-fresh sushi"
                    },
                    {
                        "name": "Sukiyabashi Jiro",
                        "rating": 4.9,
                        "price_range": "$$$$",
                        "specialty": "Premium sushi experience"
                    }
                ]
            }))
        }
        _ => Ok(json!({
            "error": "Unknown function",
            "function": tool_call.function.name
        })),
    }
}

// Type aliases to help with dynamic provider type checking
// These wouldn't be needed in practice and are only for the tool_choice example
type OpenAI = llm::backends::openai::OpenAI;
type Anthropic = llm::backends::anthropic::Anthropic;
type Google = llm::backends::google::Google;
